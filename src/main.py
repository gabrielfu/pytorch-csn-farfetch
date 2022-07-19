from __future__ import annotations
import argparse
import os
import sys
import shutil

import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn
from torchvision import transforms
from visdom import Visdom

from models import resnet18, CSN, TripletNet
from triplet_image_loader import TripletImageLoader


best_acc = 0


def main(args):
    global best_acc
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if args.visdom:
        global plotter
        plotter = VisdomLinePlotter(env_name=args.name)

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
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json',
                           args.conditions, 'train', n_triplets=args.num_train_triplets,
                           transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json',
                           args.conditions, 'val', n_triplets=args.num_val_triplets,
                           transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json',
                           args.conditions, 'test', n_triplets=args.num_test_triplets,
                           transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    backbone = resnet18(pretrained=True, embedding_size=args.dim_embed).to(device)
    csn = CSN(backbone=backbone, n_conditions=len(args.conditions),
                                  embedding_size=args.dim_embed, learned_mask=args.learned, prein=args.prein).to(device)
    global mask_var
    mask_var = csn.masks.weight
    tnet = TripletNet(csn).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    torch.backends.cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    parameters = filter(lambda p: p.requires_grad, tnet.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.test:
        test_acc = test(test_loader, tnet, criterion, 1)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(val_loader, tnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)


def train(train_loader, tnet, criterion, optimizer, device, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    mask_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3, c) in enumerate(train_loader):
        data1, data2, data3, c = data1.to(device), data2.to(device), data3.to(device), c.to(device)

        # compute output
        dista, distb, mask_norm, embed_norm, mask_embed_norm = tnet(data1, data2, data3, c)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1).to(device)

        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embed_norm / np.sqrt(data1.size(0))
        loss_mask = mask_norm / data1.size(0)
        loss = loss_triplet + args.embed_loss * loss_embedd + args.mask_loss * loss_mask

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data.item(), data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data.item())
        mask_norms.update(loss_mask.data.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg,
                       100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

    # log avg values to visdom
    if args.visdom:
        plotter.plot('acc', 'train', epoch, accs.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)
        plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)
        plotter.plot('mask_norms', 'train', epoch, mask_norms.avg)
        if epoch % 10 == 0:
            plotter.plot_mask(torch.nn.functional.relu(mask_var).data.cpu().numpy().T, epoch)


def test(test_loader, tnet, criterion, device, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    accs_cs = {}
    for condition in conditions:
        accs_cs[condition] = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3, c) in enumerate(test_loader):
        data1, data2, data3, c = data1.to(device), data2.to(device), data3.to(device), c.to(device)
        c_test = c

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3, c)
        target = torch.FloatTensor(dista.size()).fill_(1).to(device)
        test_loss = criterion(dista, distb, target).data.item()

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        for condition in conditions:
            accs_cs[condition].update(accuracy_id(dista, distb, c_test, condition), data1.size(0))
        losses.update(test_loss, data1.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    if args.visdom:
        for condition in conditions:
            plotter.plot('accs', 'acc_{}'.format(condition), epoch, accs_cs[condition].avg)
        plotter.plot(args.name, args.name, epoch, accs.avg, env='overview')
        plotter.plot('acc', 'test', epoch, accs.avg)
        plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


class VisdomLinePlotter(object):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    if args.visdom:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum() * 1.0 / dista.size()[0]


def accuracy_id(dista, distb, c, c_id):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return ((pred > 0) * (c.cpu().data == c_id)).sum() * 1.0 / (c.cpu().data == c_id).sum()


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
    parser.add_argument('--embed-loss', type=float, default=5e-3,
                        help='parameter for loss for embedding norm')
    parser.add_argument('--mask-loss', type=float, default=5e-4,
                        help='parameter for loss for mask norm')
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
    parser.add_argument('--conditions', nargs='*', type=int, default=[0, 1, 2, 3],
                        help='set of similarity notions')
    args = parser.parse_args()

    main(args)
