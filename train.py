import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import CSRNet, make_layers

import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')


def main():
    global args, best_prec1

    best_prec1 = 1e6
    # 初始化网络参数
    args = parser.parse_args()
    args.original_lr = 1e-6     # 初始学习率
    args.lr = 1e-6              # 优化器学习率
    args.batch_size = 1         # 训练集中选择的一组样本用于更新权重，1batch包含的样本数目，这里一组样本只有1个样本
    args.momentum = 0.95        # 动量
    args.decay = 5 * 1e-4       # 衰减权重
    args.start_epoch = 0        # 起始代
    args.epochs = 400            # 代数 原始为400
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    with open(args.train_json, 'r',encoding='utf-8') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r',encoding='utf-8') as outfile:
        val_list = json.load(outfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    model = CSRNet()

    model = model.cuda()

    #criterion = nn.MSELoss(size_average=False).cuda()
    criterion = nn.MSELoss(reduction='sum').cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, # 使用优化器
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']  # 迭代代数
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        # 确保重新初始化 seen 属性
            model.seen = getattr(model, 'seen', 0)
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)  # 每一代训练时更新权重参数
        torch.cuda.empty_cache()
        train(train_list, model, criterion, optimizer, epoch)   # 传参进train方法进行训练
        prec1 = validate(val_list, model, criterion)    # 传参进validate方法进行预测

        is_best = prec1 < best_prec1    # 获取最小误差
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        # 保存每代的训练状态，不论是否是最佳更新状态均进行保存
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)


def train(train_list, model, criterion, optimizer, epoch):
    # torch.cuda.empty_cache()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=model.seen,
                            
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    model.seen=getattr(model, 'seen', 0),  # 确保 seen 存在
    model.seen += len(train_loader.dataset)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        # del output
        # torch.cuda.empty_cache()


def validate(val_list, model, criterion):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 使用1 + y确保不会改变特征的量级
        return x * (1 + y.expand_as(x))


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        
        # 添加轻量级注意力
        self.attention = ChannelAttention(512)
        
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.attention(x)  # 添加注意力但保持特征量级
        x = self.backend(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    main()
