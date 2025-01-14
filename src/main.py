from __future__ import annotations
import argparse
import os
import shutil
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import resnet18, CSN, TripletNet
from triplet_dataset import TripletDataset
from metrics import AverageMeter, accuracy, accuracy_id


def main(args):
    print(f"Run: {args.name}")

    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    print(f"Device: {device}")

    # tensorboard
    log_dir = os.path.join(args.log_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log dir: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir) if args.tensorboard else None
    print(f"Tensorboard: {args.tensorboard}")

    # dataset
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    train_dataset = TripletDataset(root=args.data_dir,
                                   condition_indices=args.conditions,
                                   split="train",
                                   n_triplets=args.train_triplets,
                                   transform=transform)
    val_dataset = TripletDataset(root=args.data_dir,
                                 condition_indices=args.conditions,
                                 split="val",
                                 n_triplets=args.val_triplets,
                                 transform=transform)
    test_dataset = TripletDataset(root=args.data_dir,
                                  condition_indices=args.conditions,
                                  split="test",
                                  n_triplets=args.test_triplets,
                                  transform=transform)
    conditions = train_dataset.conditions
    condition_indices = train_dataset.condition_indices

    # data loader
    kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # model
    backbone = resnet18(pretrained=True, embedding_size=args.embed_dim).to(device)
    csn = CSN(backbone=backbone,
              n_conditions=len(args.conditions),
              embedding_size=args.embed_dim,
              learned_mask=args.learned,
              prein=args.prein).to(device)
    model = TripletNet(csn).to(device)

    # criterion & optimizer
    lr = args.lr
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    torch.backends.cudnn.benchmark = True

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('Number of params: {}'.format(n_parameters))

    # optionally resume from a checkpoint
    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    if args.test:
        test(model, test_loader, conditions, condition_indices, criterion, device, 0, None)
        return

    # training loop
    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        lr = adjust_learning_rate(lr, optimizer, epoch, writer)
        # train for one epoch
        train(model, train_loader, criterion, optimizer, device, epoch, args.embed_loss_coeff, args.mask_loss_coeff, args.log_interval, writer)
        # evaluate on validation set
        acc = test(model, val_loader, conditions, condition_indices, criterion, device, epoch, writer)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }, is_best, log_dir)

        # plot mask distribution along the embedding dimensions
        if writer is not None and epoch % 2 == 0:
            weights = F.relu(model.csn.masks.weight).data.cpu().numpy().T
            plot_condition_masks(writer, epoch, conditions, weights)


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
        writer: Optional[SummaryWriter]=None,
):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    mask_norms = AverageMeter()

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
        accs.update(acc.data.item(), data1.size(0))
        emb_norms.update(loss_embed.data.item())
        mask_norms.update(loss_mask.data.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == len(train_loader):
            print(f'{datetime.datetime.now()} \t'
                  f'Train Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)}]\t'
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) \t'
                  f'Acc: {accs.val * 100:.2f}% ({accs.avg * 100:.2f}%) \t'
                  f'Emb_Norm: {emb_norms.val:.2f} ({emb_norms.avg:.2f})')

    # log avg values
    if writer is not None:
        writer.add_scalar('Metrics/Accuracy/train', accs.avg, epoch)
        writer.add_scalar('Metrics/Loss/train', losses.avg, epoch)
        writer.add_scalar('Model/Embedding Norms/train', emb_norms.avg, epoch)
        writer.add_scalar('Model/Mask Norms/train', mask_norms.avg, epoch)
        writer.flush()


def test(
        model: TripletNet,
        test_loader: DataLoader,
        conditions: List[str],
        condition_indices: List[int],
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        writer: Optional[SummaryWriter]=None,
):
    losses = AverageMeter()
    accs = AverageMeter()
    accs_cs = {}
    for condition in conditions:
        accs_cs[condition] = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, c) in enumerate(test_loader):
            data1, data2, data3, c = data1.to(device), data2.to(device), data3.to(device), c.to(device)

            # compute output
            dist_a, dist_b, _, _, _ = model(data1, data2, data3, c)
            target = torch.FloatTensor(dist_a.size()).fill_(1).to(device)
            test_loss = criterion(dist_a, dist_b, target)

            # measure accuracy and record loss
            acc = accuracy(dist_a, dist_b)
            accs.update(acc.data.item(), data1.size(0))
            for condition, idx in zip(conditions, condition_indices):
                cond_acc = accuracy_id(dist_a, dist_b, c, idx).data.item()
                if not np.isnan(cond_acc):
                    accs_cs[condition].update(cond_acc, data1.size(0))
            losses.update(test_loss.data.item(), data1.size(0))

    print(f'{datetime.datetime.now()} \tTest set: Loss: {losses.avg:.4f}, Accuracy: {accs.avg * 100:.2f}%')
    if writer is not None:
        writer.add_scalar('Metrics/Accuracy/test', accs.avg, epoch)
        writer.add_scalar('Metrics/Loss/test', losses.avg, epoch)
        for condition in conditions:
            writer.add_scalar(f'Supplementary/Conditional Accuracy/{condition}', accs_cs[condition].avg, epoch)
        writer.flush()

    return accs.avg


def save_checkpoint(
        state: Dict,
        is_best: bool,
        log_dir: str,
):
    """Saves checkpoint to disk"""
    epoch = state["epoch"]
    filename = os.path.join(log_dir, f"checkpoint-{epoch:05d}.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(log_dir, 'model_best.pth.tar'))


def plot_condition_masks(writer: Optional[SummaryWriter], epoch: int, conditions: List[str], weights: np.ndarray):
    """
    Args:
        writer (Optional[SummaryWriter]): Tensorboard writer
        epoch (int): Epoch number
        conditions (List[str]): List of condition names
        weights (np.ndarray): (C x D) array, where C is number of conditions and D is embedding dimension
    """
    if writer is None:
        return

    # plot the bar chart
    fig, axes = plt.subplots(len(conditions), 1, figsize=(10, 8))
    x = np.arange(weights.shape[1])
    for i, condition in enumerate(conditions):
        axes[i].bar(x, weights[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_ylabel(condition, rotation=0, labelpad=30)
    axes[-1].set_xlabel("embedding dimension index")
    fig.tight_layout(pad=0)
    plt.subplots_adjust(wspace=None, hspace=None)

    # tensorboard
    writer.add_figure("Mask/Distribution", fig, epoch)
    writer.flush()


def adjust_learning_rate(
        lr: float,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        writer: Optional[SummaryWriter]=None,
):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * ((1 - 0.015) ** epoch)
    if writer is not None:
        writer.add_scalar('Model/Learning Rate', lr, epoch)
        writer.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='PyTorch Conditional Similarity Network')
    parser.add_argument('--data-dir', default="data/farfetch", metavar="N",
                        help='directory to the training data (default: "data/farfetch")')
    parser.add_argument('--batch-size', type=int, default=512, metavar="N",
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=30, metavar="N",
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--start-epoch', type=int, default=1, metavar="N",
                        help='number of start epoch (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar="x",
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--seed', type=int, default=42, metavar="N",
                        help='random seed (default: 42)')
    parser.add_argument('--cuda', default=True, action='store_true',
                        help='enables CUDA training (default: True)')
    parser.add_argument('--log-interval', type=int, default=20, metavar="N",
                        help='how many batches to wait before logging training status (default: 20)')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default=f"csn_{now}", type=str,
                        help='name of experiment (default: "csn_yyyymmdd_hhmmss")')
    parser.add_argument('--embed-loss-coeff', type=float, default=5e-3, metavar="x",
                        help='loss coefficient for embedding norm (default: 5e-3)')
    parser.add_argument('--mask-loss-coeff', type=float, default=5e-4, metavar="x",
                        help='loss coefficient for mask norm (default: 5e-4)')
    parser.add_argument('--train-triplets', type=int, default=100000, metavar="N",
                        help='how many unique training triplets (default: 100000)')
    parser.add_argument('--val-triplets', type=int, default=20000, metavar="N",
                        help='how many unique validation triplets (default: 20000)')
    parser.add_argument('--test-triplets', type=int, default=40000, metavar="N",
                        help='how many unique test triplets (default: 40000)')
    parser.add_argument('--embed-dim', type=int, default=64, metavar="N",
                        help='embedding dimension (default: 64)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='to only run inference on test set (default: False)')
    parser.add_argument('--learned', default=True, action='store_true',
                        help='to learn masks from random initialization (default: True)')
    parser.add_argument('--prein', default=False, action='store_true',
                        help='to initialize masks to be disjoint (default: False)')
    parser.add_argument('--tensorboard', default=False, action='store_true',
                        help='use tensorboard to track and plot (default: False)')
    parser.add_argument('--log-dir', default="runs",
                        help='directory to save checkpoints & tensorboard data (default: "runs")')
    parser.add_argument('--conditions', nargs='*', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='set of similarity notions')
    args = parser.parse_args()

    main(args)
