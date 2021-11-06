import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# backbone nets
from models.backbones.resnet38d_v2 import ResNet38
from models.backbones.vgg16d import VGG16
from models.backbones.resnets import ResNet101, ResNet50

# modules
from models.mods import ASPP
from models.mods import PAMR
from models.mods import StochasticGate
from models.mods import GCI
from models.mods import SpatialAttention, ChannelAttention

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def focal_loss(x, p=1, c=0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)

def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x

def pseudo_gtmask(mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
    """Convert continuous mask into binary mask"""
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max[:, :1] *= 0.7
    mask_max[:, 1:] *= cutoff_top
    # mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
    pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs, c, h, w)


def balanced_mask_loss_ce(mask, pseudo_gt, gt_labels, ignore_index=255):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs, c, h, w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs, c, -1).sum(-1)
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
    class_weight = (pseudo_gt * class_weight[:, :, None, None]).sum(1).view(bs, -1)

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss = loss.view(bs, -1)

    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1  # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss


def network_CAM_CASA_WGAP_PCM(cfg):
    if cfg.BACKBONE == "resnet38":
        print("Backbone: ResNet38")
        backbone = ResNet38
    elif cfg.BACKBONE == "vgg16":
        print("Backbone: VGG16")
        backbone = VGG16
    elif cfg.BACKBONE == "resnet50":
        print("Backbone: ResNet50")
        backbone = ResNet50
    elif cfg.BACKBONE == "resnet101":
        print("Backbone: ResNet101")
        backbone = ResNet101
    else:
        raise NotImplementedError("No backbone found for '{}'".format(cfg.BACKBONE))

    '''
            CAM+SpatialAttention+CA+WGAP
    '''
    class CAM_SpatialAttention_CASA_WGAP_PCM(backbone):

        def __init__(self, config, pre_weights=None, num_classes=21, dropout=True):
            super().__init__()

            self.cfg = config
            self.num_classes = num_classes

            self.fc8 = nn.Conv2d(self.fan_out(), num_classes, 1, bias=False)
            self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
            self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
            self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)

            nn.init.xavier_uniform_(self.fc8.weight)
            torch.nn.init.xavier_uniform_(self.fc8.weight)
            torch.nn.init.kaiming_normal_(self.f8_3.weight)
            torch.nn.init.kaiming_normal_(self.f8_4.weight)
            torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)



            cls_modules = [self.fc8]
            if dropout:
                cls_modules.insert(0, nn.Dropout2d(0.5))

            self.caatention = ChannelAttention(in_planes=4096)
            self.attention = SpatialAttention(kernel_size=7)
            self.cls_branch = nn.Sequential(*cls_modules)
            self.mask_branch = nn.Sequential(self.fc8, nn.ReLU())

            self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8]
            if pre_weights:
                self._init_weights(pre_weights)
            self._mask_logits = None

            self._fix_running_stats(self, fix_params=True)  # freeze backbone BNs

        def forward_backbone(self, x):
            self._mask_logits = super().forward(x)
            return self._mask_logits

        def forward(self, y, y_raw=None, labels=None):
            test_mode = labels is None

            d = self.forward_backbone(y)
            x = d['conv6']
            Channel_attention = self.caatention(x)

            x = torch.mul(x, Channel_attention)
            Spatial_weight, attention_map = self.attention(x)

            x = torch.mul(x, Spatial_weight)

            x = self.cls_branch(x)
            bs, c, h, w = x.size()
            with torch.no_grad():
                cam_d = F.relu(x.detach())
                cam_d_max = torch.max(cam_d.view(bs, c, -1), dim=-1)[0].view(bs, c, 1, 1) + 1e-5
                cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
                cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
                cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
                cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0

            f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
            f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
            x_s = F.interpolate(y, (h, w), mode='bilinear', align_corners=True)
            f = torch.cat([x_s, f8_3, f8_4], dim=1)
            cam_rv = self._rescale_and_clean(self.PCM(cam_d_norm, f), y, labels)
            masks_dec = F.softmax(cam_rv, dim=1)


            masks = F.softmax(x, dim=1)
            # reshaping
            features = x.view(bs, c, -1)
            masks_ = masks.view(bs, c, -1)

            # classification loss
            cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))

            # focal penalty loss
            cls_2 = focal_loss(masks_.mean(-1), \
                               p=self.cfg.FOCAL_P, \
                               c=self.cfg.FOCAL_LAMBDA)

            # adding the losses together
            cls = cls_1[:, 1:] + cls_2[:, 1:]

            if test_mode:
                return cls, rescale_as(masks, y)

            self._mask_logits = x

            # foreground stats
            b, c, h, w = masks.size()
            masks_ = masks.view(b, c, -1)
            masks_ = masks_[:, 1:]
            cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

            # mask refinement with PAMR
            # masks_dec = self.run_pamr(y_raw, masks.detach())

            # upscale the masks & clean
            masks = self._rescale_and_clean(masks, y, labels)
            masks_dec = self._rescale_and_clean(masks_dec, y, labels)

            # create pseudo GT
            pseudo_gt = pseudo_gtmask(masks_dec).detach()
            loss_mask = balanced_mask_loss_ce(self._mask_logits, pseudo_gt, labels)

            return cls, cls_fg, {"cam": masks, "dec": masks_dec}, self._mask_logits, pseudo_gt, loss_mask, None

        def _rescale_and_clean(self, masks, image, labels):
            masks = F.interpolate(masks, size=image.size()[-2:], mode='bilinear', align_corners=True)
            masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
            return masks

        def PCM(self, cam, f):
            n, c, h, w = f.size()
            cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
            f = self.f9(f)
            f = f.view(n, -1, h * w)
            f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

            aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
            aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
            cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

            return cam_rv

    return CAM_SpatialAttention_CASA_WGAP_PCM