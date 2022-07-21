from __future__ import annotations
import argparse
import os
import sys
import shutil
import datetime
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn
from torchvision import transforms
from visdom import Visdom

from models import resnet18, CSN, TripletNet
from triplet_dataset import TripletDataset
from metrics import AverageMeter, accuracy, accuracy_id

plotter: VisdomLinePlotter

def main(args):
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if args.visdom:
        plotter = VisdomLinePlotter(env_name=args.name)

    # data loader
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(
        TripletDataset(root='./data/farfetch',
                       condition_indices=args.conditions,
                       split="train",
                       n_triplets=args.num_train_triplets,
                       transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(
        TripletDataset(root='./data/farfetch',
                       condition_indices=args.conditions,
                       split="val",
                       n_triplets=args.num_val_triplets,
                       transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = DataLoader(
        TripletDataset(root='./data/farfetch',
                       condition_indices=args.conditions,
                       split="test",
                       n_triplets=args.num_test_triplets,
                       transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    # model
    backbone = resnet18(pretrained=True, embedding_size=args.dim_embed).to(device)
    csn = CSN(backbone=backbone,
              n_conditions=len(args.conditions),
              embedding_size=args.dim_embed,
              learned_mask=args.learned,
              prein=args.prein).to(device)
    model = TripletNet(csn).to(device)

    # optionally resume from a checkpoint
    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # criterion & optimizer
    lr = args.lr
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    torch.backends.cudnn.benchmark = True

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.test:
        test(model, test_loader, args.conditions, criterion, device, 0, False, None)
        return

    # training loop
    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        lr = adjust_learning_rate(lr, optimizer, epoch, args.visdom)
        # train for one epoch
        train(model, train_loader, criterion, optimizer, device, epoch, args.embed_loss_coeff, args.mask_loss_coeff, args.log_interval, args.visdom)
        # evaluate on validation set
        acc = test(model, val_loader, args.conditions, criterion, device, epoch, args.visdom, args.name)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
        }, is_best, args.name)

        # plot mask distribution along the embedding dimensions
        if args.visdom and epoch % 10 == 0:
            plotter.plot_mask(F.relu(model.csn.masks.weight).data.cpu().numpy().T, epoch)


def train(
        model: TripletNet,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        embed_loss_coeff: float,
        mask_loss_coeff: float,
        print_every: int=10,
        visdom: bool=False
):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    mask_norms = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (data1, data2, data3, c) in enumerate(train_loader):
        data1, data2, data3, c = data1.to(device), data2.to(device), data3.to(device), c.to(device)

        # compute output
        dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm = model(data1, data2, data3, c)
        # 1 means, dist_a should be larger than dist_b
        target = torch.FloatTensor(dist_a.size()).fill_(1).to(device)

        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embed = embed_norm / np.sqrt(data1.size(0))
        loss_mask = mask_norm / data1.size(0)
        loss = loss_triplet + embed_loss_coeff * loss_embed + mask_loss_coeff * loss_mask

        # measure accuracy and record loss
        acc = accuracy(dist_a, dist_b)
        losses.update(loss_triplet.data.item(), data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embed.data.item())
        mask_norms.update(loss_mask.data.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % print_every == 0:
            print(f'{datetime.datetime.now()}'
                  f'Train Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)}]\t'
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) \t'
                  f'Acc: {accs.val * 100:.2f}% ({accs.avg * 100:.2f}%) \t'
                  f'Emb_Norm: {emb_norms.val:.2f} ({emb_norms.avg:.2f})')

    # log avg values to visdom
    if visdom:
        plotter.plot('acc', 'train', epoch, accs.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)
        plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)
        plotter.plot('mask_norms', 'train', epoch, mask_norms.avg)


def test(
        model: TripletNet,
        test_loader: DataLoader,
        conditions: List[str],
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        visdom: bool=False,
        run_name: str=None,
):
    losses = AverageMeter()
    accs = AverageMeter()
    accs_cs = {}
    for condition in conditions:
        accs_cs[condition] = AverageMeter()

    # switch to evaluation mode
    model.eval()
    for batch_idx, (data1, data2, data3, c) in enumerate(test_loader):
        data1, data2, data3, c = data1.to(device), data2.to(device), data3.to(device), c.to(device)
        c_test = c

        # compute output
        dist_a, dist_b, _, _, _ = model(data1, data2, data3, c)
        target = torch.FloatTensor(dist_a.size()).fill_(1).to(device)
        test_loss = criterion(dist_a, dist_b, target).data.item()

        # measure accuracy and record loss
        acc = accuracy(dist_a, dist_b)
        accs.update(acc, data1.size(0))
        for condition in conditions:
            accs_cs[condition].update(accuracy_id(dist_a, dist_b, c_test, condition), data1.size(0))
        losses.update(test_loss, data1.size(0))

    print(f'Test set: Average loss: {losses.avg:.4f}, Accuracy: {accs.avg * 100:.2f}%\n')
    if visdom:
        for condition in conditions:
            plotter.plot('accs', 'acc_{}'.format(condition), epoch, accs_cs[condition].avg)
        plotter.plot(run_name, run_name, epoch, accs.avg, env='overview')
        plotter.plot('acc', 'test', epoch, accs.avg)
        plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg


def save_checkpoint(
        state: Dict,
        is_best: bool,
        run_name: str,
):
    """Saves checkpoint to disk"""
    directory = f"runs/{run_name}"
    os.makedirs(directory, exist_ok=True)
    epoch = state["epoch"]
    filename = os.path.join(directory, f"checkpoint-{epoch:05d}.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))


class VisdomLinePlotter:
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name],
                                 name=split_name)

    def plot_mask(self, masks, epoch):
        self.viz.bar(
            X=masks,
            env=self.env,
            opts=dict(
                stacked=True,
                title=epoch,
            )
        )


def adjust_learning_rate(
        lr: float,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        visdom: bool,
):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * ((1 - 0.015) ** epoch)
    if visdom:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Conditional Similarity Network')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='number of start epoch (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--cuda', default=True, action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                        help='name of experiment')
    parser.add_argument('--embed-loss-coeff', type=float, default=5e-3,
                        help='loss coefficient for embedding norm')
    parser.add_argument('--mask-loss-coeff', type=float, default=5e-4,
                        help='loss coefficient for mask norm')
    parser.add_argument('--num-train-triplets', type=int, default=100000,
                        help='how many unique training triplets (default: 100000)')
    parser.add_argument('--num-val-triplets', type=int, default=20000,
                        help='how many unique validation triplets (default: 20000)')
    parser.add_argument('--num-test-triplets', type=int, default=40000,
                        help='how many unique test triplets (default: 40000)')
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='embedding dimension (default: 64)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='to only run inference on test set (default: False)')
    parser.add_argument('--learned', default=True, action='store_true',
                        help='to learn masks from random initialization (default: True)')
    parser.add_argument('--prein', default=False, action='store_true',
                        help='to initialize masks to be disjoint (default: False)')
    parser.add_argument('--visdom', default=False, action='store_true',
                        help='use visdom to track and plot (default: False)')
    parser.add_argument('--conditions', nargs='*', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='set of similarity notions')
    args = parser.parse_args()

    main(args)
