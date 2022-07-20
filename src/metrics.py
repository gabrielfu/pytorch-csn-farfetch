class AverageMeter:
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


def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum() * 1.0 / dista.size()[0]


def accuracy_id(dista, distb, c, c_id):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return ((pred > 0) * (c.cpu().data == c_id)).sum() * 1.0 / (c.cpu().data == c_id).sum()