import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# backbone nets
from models.backbones.resnet38d import ResNet38
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

def network_CAM_CASA_WGAP_v3(cfg):
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
    class CAM_SpatialAttention_CASA_WGAP_v3(backbone):

        def __init__(self, config, pre_weights=None, num_classes=21, dropout=True):
            super().__init__()

            self.cfg = config
            self.num_classes = num_classes

            self.fc8 = nn.Conv2d(self.fan_out(), num_classes, 1, bias=False)
            nn.init.xavier_uniform_(self.fc8.weight)

            cls_modules = [self.fc8]
            if dropout:
                cls_modules.insert(0, nn.Dropout2d(0.5))

            self.caatention = ChannelAttention(in_planes=4096)
            self.attention = SpatialAttention(kernel_size=7)
            self.cls_branch = nn.Sequential(*cls_modules)
            self.mask_branch = nn.Sequential(self.fc8, nn.ReLU())

            self.from_scratch_layers = [self.fc8]
            if pre_weights:
                self._init_weights(pre_weights)
            self._mask_logits = None

            self._fix_running_stats(self, fix_params=True)  # freeze backbone BNs

        def forward_backbone(self, x):
            self._mask_logits = super().forward(x)
            return self._mask_logits

        def forward(self, y, _=None, labels=None):
            test_mode = labels is None

            x = self.forward_backbone(y)
            Channel_attention = self.caatention(x)

            x = torch.mul(x, Channel_attention)
            Spatial_weight, attention_map = self.attention(x)

            x = torch.mul(x, Spatial_weight)

            x = self.cls_branch(x)
            # constant BG scores
            # bg = torch.ones_like(x[:, :1])
            # x = torch.cat([bg, x], 1)

            bs, c, h, w = x.size()
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

            # foreground stats
            b, c, h, w = masks.size()
            masks_ = masks.view(b, c, -1)
            masks_ = masks_[:, 1:]
            cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

            # upscale the masks & clean
            pseudo_gt = rescale_as(masks, y)
            masks = self._rescale_and_clean(masks, y, labels)
            mask_logits = rescale_as(x, y) #x
            # attention loss
            loss_at = torch.sum(attention_map.pow(2), dim=-1)
            return cls, cls_fg, {"cam": masks}, mask_logits, pseudo_gt, None, loss_at

        def _rescale_and_clean(self, masks, image, labels):
            masks = F.interpolate(masks, size=image.size()[-2:], mode='bilinear', align_corners=True)
            masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
            return masks

    return CAM_SpatialAttention_CASA_WGAP_v3