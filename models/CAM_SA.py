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
from models.mods import SpatialAttention,ChannelAttention

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def network_CAM_SA(cfg):
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
       CAM+SpatialAttention
    '''
    class CAM_SpatialAttention(backbone):

        def __init__(self, config, pre_weights=None, num_classes=21, dropout=True):
            super().__init__()

            self.cfg = config

            self.fc8 = nn.Conv2d(self.fan_out(), num_classes - 1, 1, bias=False)
            nn.init.xavier_uniform_(self.fc8.weight)

            cls_modules = [nn.AdaptiveAvgPool2d((1, 1)), self.fc8, Flatten()]
            if dropout:
                cls_modules.insert(0, nn.Dropout2d(0.5))

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

        def forward_cls(self, x):
            return self.cls_branch(x)

        def forward_mask(self, x, size):
            logits = self.fc8(x)
            masks = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
            masks = F.relu(masks)

            # CAMs are unbounded
            # so let's normalised it first
            # (see jiwoon-ahn/psa)
            b, c, h, w = masks.size()
            masks_ = masks.view(b, c, -1)
            z, _ = masks_.max(-1, keepdim=True)
            masks_ /= (1e-5 + z)
            masks = masks.view(b, c, h, w)

            bg = torch.ones_like(masks[:, :1])
            masks = torch.cat([self.cfg.BG_SCORE * bg, masks], 1)

            # note, that the masks contain the background as the first channel
            return logits, masks

        def forward(self, y, _=None, labels=None):
            test_mode = labels is None

            x = self.forward_backbone(y)
            Spatial_weight, attention_map = self.attention(x)

            x = torch.mul(x, Spatial_weight)
            cls = self.forward_cls(x)
            logits, masks = self.forward_mask(x, y.size()[-2:])

            if test_mode:
                return cls, masks

            # foreground stats
            b, c, h, w = masks.size()
            masks_ = masks.view(b, c, -1)
            masks_ = masks_[:, 1:]
            cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

            # upscale the masks & clean
            masks = self._rescale_and_clean(masks, y, labels)

            # attention loss
            loss_at = torch.sum(attention_map.pow(2), dim=-1)

            return cls, cls_fg, {"cam": masks}, logits, None, None, loss_at

        def _rescale_and_clean(self, masks, image, labels):
            masks = F.interpolate(masks, size=image.size()[-2:], mode='bilinear', align_corners=True)
            masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
            return masks

    return CAM_SpatialAttention