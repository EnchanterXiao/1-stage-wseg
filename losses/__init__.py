import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class MLHingeLoss(nn.Module):

    def forward(self, x, y, reduction='mean'):
        """
            y: labels have standard {0,1} form and will be converted to indices
        """
        b, c = x.size()
        idx = (torch.arange(c) + 1).type_as(x)
        y_idx, _ = (idx * y).sort(-1, descending=True)
        y_idx = (y_idx - 1).long()

        return F.multilabel_margin_loss(x, y_idx, reduction=reduction)

def get_criterion(loss_name, **kwargs):

    losses = {
            "SoftMargin": nn.MultiLabelSoftMarginLoss,
            "Hinge": MLHingeLoss
            }

    return losses[loss_name](**kwargs)


#
# Mask self-supervision
#
def mask_loss_ce(mask, pseudo_gt, ignore_index=255):
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    weight = pseudo_gt.sum(1).type_as(mask_gt)
    mask_gt += (1 - weight) * ignore_index

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index)
    return loss.mean()


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
