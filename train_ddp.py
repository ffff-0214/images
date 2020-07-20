import argparse
import json
import logging
import os
import sys
import time
import math

# from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

import apex
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel as DDP

from model.mixup import mixup_criterion, mixup_data
from data.DegreesData import DegreesData  # noqa
from model.utils import (AverageMeter, ConfusionMatrix,
                         plot_confusion_matrix)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', '-c', default='./configs/efficientnet.json', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--ckpt_path_save', '-ckpt_s',
                    default='/data/gukedata/ckpt/', help='checkpoint path to save')
parser.add_argument('--log_path', '-lp',
                    default='/data/gukedata/log/', help='log path')
parser.add_argument('--num_workers', default=12, type=int,
                    help='number of workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1,2,3', type=str, help='comma separated indices of GPU to use,'
                    ' e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')
parser.add_argument('--start_epoch', '-s', default=0,
                    type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=500,
                    type=int, help='end epoch')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument(
    '--ckpt', '-ckpt', default='/data/gukedata/ckpt/pre_efficientnet_b7_1/', help='checkpoint path')
parser.add_argument('--experiment_id', '-eid',
                    default='0', help='experiment id')
parser.add_argument('--experiment_name', '-name',
                    default='pre_efficientnet_b7', help='experiment name')
parser.add_argument('--alpha', default=0, type=float,
                    help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[60, 100, 120], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument("--local_rank", default=0, type=int)

use_cuda = True
args = parser.parse_args()
device = torch.device("cuda" if use_cuda else "cpu")


if args.local_rank == 0:
    log_path = os.path.join(
        args.log_path, args.experiment_name + "_" + str(args.experiment_id))
    print("log_path:", log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    ckpt_path_save = os.path.join(
        args.ckpt_path_save, args.experiment_name + "_" + str(args.experiment_id))
    if not os.path.exists(ckpt_path_save):
        os.makedirs(ckpt_path_save)
    print("ckpt_path_save:", ckpt_path_save)
    log_path_cm = os.path.join(
        log_path, 'confusion_matrix')
    if not os.path.exists(log_path_cm):
        os.makedirs(log_path_cm)


def load_checkpoint(args, net, optimizer, amp):
    print("Use ckpt: ", args.ckpt)
    assert len(args.ckpt) != 0, "Please input a valid ckpt_path"
    # checkpoint = torch.load(args.ckpt)
    checkpoint = torch.load(
        args.ckpt, map_location=lambda storage, loc: storage.cuda(args.gpu))
    pretrained_dict = checkpoint['state_dict']
    net.load_state_dict(pretrained_dict)
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    return net, epoch


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def reduce_tensor(tensor, reduction=True):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if reduction:
        rt /= args.world_size
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def train_epoch(epoch, summary, summary_writer, model, loss_fn, optimizer, dataloader_train, cfg):
    model.train()
    num_classes = cfg['num_classes']
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)

    steps = len(dataloader_train)
    batch_size = dataloader_train.batch_size

    dataiter = iter(dataloader_train)
    time_now = time.time()
    loss_sum = 0
    acc_sum = 0

    summary['epoch'] = epoch

    if args.local_rank == 0:
        print("steps:", steps)
    prefetcher = data_prefetcher(dataiter)
    img, target = prefetcher.next()
    for step in range(steps):
        data = img.to(device)
        target = target.to(device)

        # mixup
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        data, target_a, target_b, lam = mixup_data(
            data, target, args.alpha, use_cuda)
        # print(data.shape)
        output = model(data)
        output = output.view(int(batch_size), num_classes)
        target = target.view(int(batch_size))
        target = target.long()

        # mixup
        loss_func = mixup_criterion(target_a, target_b, lam)
        # TODO: 实现顺序代价敏感
        loss = loss_func(loss_fn, output)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # scheduler.step()
        # lr = scheduler.get_last_lr()[0]
        probs = F.softmax(output, dim=1)
        # torch.max(a,1) 返回每一行中最大值的那个元素FloatTensor，且返回其索引LongTensor（返回最大元素在这一行的列索引）
        _, predicts = torch.max(probs, 1)

        # mixup
        acc = (lam *
               (predicts == target_a).type(torch.cuda.FloatTensor).sum() +
               (1 - lam) *
               (predicts == target_b).type(torch.cuda.FloatTensor).sum()
               ) / img.size(0)
        # acc = (predicts == target).type(torch.cuda.FloatTensor).sum().item()
        for t in range(num_classes):
            for p in range(num_classes):
                count = (predicts[target == t] == p).type(
                    torch.cuda.FloatTensor).sum()
                reduced_count = reduce_tensor(count.data, reduction=False)

                confusion_matrix.update(t, p, to_python_float(reduced_count))

        reduced_loss = reduce_tensor(loss.data)
        reduced_acc = reduce_tensor(acc.data)

        train_loss.update(to_python_float(reduced_loss))
        train_acc.update(to_python_float(reduced_acc))

        if args.local_rank == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            logging.info(
                'Epoch : {}, Step : {}, Training Loss : {:.5f}, '
                'Training Acc : {:.3f}, Run Time : {:.2f}'
                .format(
                    summary['epoch'] + 1,
                    summary['step'] + 1, train_loss.avg, train_acc.avg, time_spent))

            summary['step'] += 1

        img, target = prefetcher.next()

    if args.local_rank == 0:
        time_spent = time.time() - time_now
        time_now = time.time()
        summary_writer.add_scalar(
            'train/loss', train_loss.val, summary['epoch'] + epoch)
        summary_writer.add_scalar(
            'train/acc', train_acc.val, summary['epoch'] + epoch)
        # summary_writer.add_scalar(
        #     'learning_rate', lr, summary['step'] + steps*epoch)
        summary_writer.flush()
        summary['confusion_matrix'] = plot_confusion_matrix(
            confusion_matrix.matrix,
            cfg['labels'],
            tensor_name='train/Confusion matrix')
        # summary['loss'] = train_loss.avg
        # summary['acc'] = acc_sum / (steps * (batch_size))
        # summary['acc'] = train_acc.avg
        summary['epoch'] = epoch

    return summary


def valid_epoch(summary, summary_writer, epoch, model, loss_fn, dataloader_valid, cfg):
    model.eval()
    num_classes = cfg['num_classes']
    eval_loss = AverageMeter()
    eval_acc = AverageMeter()
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)

    dataloader = [dataloader_valid]

    name = cfg['labels']

    time_now = time.time()
    loss_sum = 0
    acc_sum = 0
    count = 0
    steps_count = 0
    for i in range(len(dataloader)):
        steps = len(dataloader[i])
        batch_size = dataloader[i].batch_size
        dataiter = iter(dataloader[i])
        # 使用 torch,no_grad()构建不需要track的上下文环境
        with torch.no_grad():
            acc_tmp = 0
            loss_tmp = 0
            prefetcher = data_prefetcher(dataiter)
            img, target = prefetcher.next()

            for step in range(steps):
                # data, target = next(dataiter)
                data = img.to(device)
                target = target.to(device)

                output = model(data)
                output = output.view(int(batch_size), num_classes)
                target = target.view(int(batch_size))
                target = target.long()
                loss = loss_fn(output, target)
                torch.cuda.synchronize()
                probs = F.softmax(output, dim=1)
                _, predicts = torch.max(probs, 1)

                for t in range(num_classes):
                    for p in range(num_classes):
                        count = (predicts[target == t] == p).type(
                            torch.cuda.FloatTensor).sum()
                        reduced_count = reduce_tensor(
                            count.data, reduction=False)

                        confusion_matrix.update(t, p,
                                                to_python_float(reduced_count))

                acc = (predicts == target).type(
                    torch.cuda.FloatTensor).sum() * 1.0 / img.size(0)

                reduced_loss = reduce_tensor(loss.data)
                reduced_acc = reduce_tensor(acc.data)

                eval_loss.update(to_python_float(reduced_loss))
                eval_acc.update(to_python_float(reduced_acc))

                if args.local_rank == 0:
                    time_spent = time.time() - time_now
                    time_now = time.time()
                    logging.info(
                        'data_num : {}, Step : {}, Testing Loss : {:.5f}, '
                        'Testing Acc : {:.3f}, Run Time : {:.2f}'
                        .format(
                            str(i),
                            summary['step'] + 1, reduced_loss, reduced_acc, time_spent))
                    summary['step'] += 1

                img, target = prefetcher.next()

    if args.local_rank == 0:
        summary['confusion_matrix'] = plot_confusion_matrix(
            confusion_matrix.matrix,
            cfg['labels'],
            tensor_name='train/Confusion matrix')
        summary['loss'] = eval_loss.avg
        # summary['acc'] = acc_sum / (steps * (batch_size))
        summary['acc'] = eval_acc.avg

    return summary


def adjust_learning_rate(optimizer, epoch, cfg, args):
    """Decay the learning rate based on schedule"""
    lr = cfg['lr']
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.end_epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.2 if epoch >= milestone else 1.

    optimizer.param_groups[0]['lr'] = lr
    for param_group in optimizer.param_groups:
        print("param_group lr: ", param_group['lr'])

    return lr


def run():
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    # num_GPU = len(args.device_ids.split(','))
    batch_size_train = cfg['train_batch_size']
    batch_size_valid = cfg['test_batch_size']
    num_workers = args.num_workers

    model = EfficientNet.from_pretrained(
        cfg['model'], num_classes=cfg['num_classes'])

    model = apex.parallel.convert_syncbn_model(model)

    # model = DataParallel(model, device_ids=None)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.Tensor([0.5, 0.7])).to(device)

    # loss_fn = nn.CrossEntropyLoss().to(device)
    # loss_fn = [nn.CrossEntropyLoss().to(device), nn.SmoothL1Loss().to(device)]
    if cfg['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                              weight_decay=1e-4)

    elif cfg['optimizer'] == 'Adam':
        optimizer = optimizers.FusedAdam(model.parameters(),
                                         lr=cfg['lr'],
                                         betas=(0.9, 0.999),
                                         weight_decay=1e-4)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level="O1",
                                      )

    if args.resume:
        model, epoch = load_checkpoint(args, model, optimizer, amp)
        if args.start_epoch < epoch:
            args.start_epoch = epoch

    if args.distributed:
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    dataset_valid = DegreesData(
        cfg['test_data_path'], cfg['image_size'], sample=False)

    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_valid)

    dataloader_valid = DataLoader(dataset_valid,
                                  sampler=eval_sampler,
                                  batch_size=batch_size_valid,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=False)

    summary_train = {'epoch': 0, 'step': 0,
                     'fp': 0, 'tp': 0, 'Neg': 0, 'Pos': 0}
    summary_valid = {'loss': float('inf'), 'step': 0, 'acc': 0}
    summary_writer = None

    if args.local_rank == 0:
        summary_writer = SummaryWriter(log_path)

    loss_valid_best = float('inf')
    lr = cfg['lr']

    for epoch in range(args.start_epoch, args.end_epoch):
        lr = adjust_learning_rate(optimizer, epoch, cfg, args)

        dataset_train = DegreesData(
            cfg['train_data_path'], cfg['image_size'], istraining=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        dataloader_train = DataLoader(dataset_train,
                                      sampler=train_sampler,
                                      batch_size=batch_size_train,
                                      num_workers=num_workers,
                                      drop_last=True,
                                      shuffle=(train_sampler is None))
        summary_train = train_epoch(epoch, summary_train,  summary_writer, model,
                                    loss_fn, optimizer, dataloader_train, cfg)
        if args.local_rank == 0:
            if epoch % 10 == 0:
                torch.save({'epoch': summary_train['epoch'],
                            'step': summary_train['step'],
                            'state_dict': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'amp': amp.state_dict()},
                           (ckpt_path_save + '/' + str(epoch) + '.ckpt'))

            summary_writer.add_figure(
                'train/confusion matrix', summary_train['confusion_matrix'], epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            if args.local_rank == 0:
                print('Learning_rate:', lr)
            break
        # summary_writer.add_scalar(
        #   'ROC',summary_train['tp']*1.0 / summary_train['Pos'],summary_train['fp']*1.0 / summary_train['Neg'])
        if epoch % 1 == 0:
            summary_valid = valid_epoch(summary_valid, summary_writer, epoch, model, loss_fn,
                                        dataloader_valid, cfg)
            if args.local_rank == 0:
                summary_writer.add_scalar(
                    'valid/loss', summary_valid['loss'], epoch)
                summary_writer.add_scalar(
                    'valid/acc', summary_valid['acc'], epoch)

                summary_writer.add_figure(
                    'valid/confusion matrix', summary_valid['confusion_matrix'], epoch)
                summary_valid['confusion_matrix'].savefig(
                    log_path_cm+'/valid_confusion_matrix_'+str(epoch)+'.png')
        if args.local_rank == 0:
            if summary_valid['loss'] < loss_valid_best:
                loss_valid_best = summary_valid['loss']
                torch.save({'epoch': summary_train['epoch'],
                            'step': summary_train['step'],
                            'state_dict': model.module.state_dict()},
                           os.path.join(ckpt_path_save, 'best.ckpt'))
            summary_writer.flush()


def main():
    logging.basicConfig(level=logging.INFO)
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    run()


if __name__ == '__main__':
    main()
