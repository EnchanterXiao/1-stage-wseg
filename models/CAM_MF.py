import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# backbone nets
from models.backbones.resnet38d_v2 import ResNet38
from models.backbones.vgg16d import VGG16
from models.backbones.resnets import ResNet101, ResNet50

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def network_CAM_MF(cfg):
    if cfg.BACKBONE == "resnet38":
        print("Backbone: ResNet38_v2")
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

    class CAM_MF(backbone):

        def __init__(self, config, pre_weights=None, num_classes=21, dropout=True):
            super().__init__()

            self.cfg = config

            self.fc8_6 = nn.Conv2d(self.fan_out(), num_classes - 1, 1, bias=False)
            self.fc8_5 = nn.Conv2d(1024, num_classes - 1, 1, bias=False)
            self.fc8_4 = nn.Conv2d(512, num_classes - 1, 1, bias=False)
            self.fc8_3 = nn.Conv2d(256, num_classes - 1, 1, bias=False)
            nn.init.xavier_uniform_(self.fc8_6.weight)
            nn.init.xavier_uniform_(self.fc8_5.weight)
            nn.init.xavier_uniform_(self.fc8_4.weight)
            nn.init.xavier_uniform_(self.fc8_3.weight)


            cls_modules_6 = [nn.AdaptiveAvgPool2d((1, 1)), self.fc8_6, Flatten()]
            cls_modules_5 = [nn.AdaptiveAvgPool2d((1, 1)), self.fc8_5, Flatten()]
            cls_modules_4 = [nn.AdaptiveAvgPool2d((1, 1)), self.fc8_4, Flatten()]
            cls_modules_3 = [nn.AdaptiveAvgPool2d((1, 1)), self.fc8_3, Flatten()]

            if dropout:
                cls_modules_6.insert(0, nn.Dropout2d(0.5))
                cls_modules_5.insert(0, nn.Dropout2d(0.5))
                cls_modules_4.insert(0, nn.Dropout2d(0.5))
                cls_modules_3.insert(0, nn.Dropout2d(0.5))

            self.cls_branch_6 = nn.Sequential(*cls_modules_6)
            self.cls_branch_5 = nn.Sequential(*cls_modules_5)
            self.cls_branch_4 = nn.Sequential(*cls_modules_4)
            self.cls_branch_3 = nn.Sequential(*cls_modules_3)
            self.mask_branch_6 = nn.Sequential(self.fc8_6, nn.ReLU())
            self.mask_branch_5 = nn.Sequential(self.fc8_5, nn.ReLU())
            self.mask_branch_4 = nn.Sequential(self.fc8_4, nn.ReLU())
            self.mask_branch_3 = nn.Sequential(self.fc8_3, nn.ReLU())

            self.from_scratch_layers = [self.fc8]
            self._init_weights(pre_weights)
            self._mask_logits = None

            self._fix_running_stats(self, fix_params=True) # freeze backbone BNs

        def forward_backbone(self, x):
            self._mask_logits = super().forward(x)
            return self._mask_logits

        def forward_cls(self, x):
            return (self.cls_branch_6(x["conv6"]) + self.cls_branch_6(x["conv5"])
                    + self.cls_branch_6(x["conv4"]) + self.cls_branch_6(x["conv3"]))/4

        def forward_mask(self, x, size):
            logits_6 = self.fc8_6(x["conv6"])
            masks = F.interpolate(logits_6, size=size, mode='bilinear', align_corners=True)
            masks = F.relu(masks)

            logits_5 = self.fc8_6(x["conv5"])
            masks = F.interpolate(logits_5, size=size, mode='bilinear', align_corners=True)
            masks += F.relu(masks)

            logits_4 = self.fc8_6(x["conv4"])
            masks = F.interpolate(logits_4, size=size, mode='bilinear', align_corners=True)
            masks += F.relu(masks)

            logits_3 = self.fc8_6(x["conv3"])
            masks = F.interpolate(logits_3, size=size, mode='bilinear', align_corners=True)
            masks += F.relu(masks)

            # CAMs are unbounded
            # so let's normalised it first
            # (see jiwoon-ahn/psa)
            b,c,h,w = masks.size()
            masks_ = masks.view(b,c,-1)
            z, _ = masks_.max(-1, keepdim=True)
            masks_ /= (1e-5 + z)
            masks = masks.view(b,c,h,w)

            bg = torch.ones_like(masks[:, :1])
            masks = torch.cat([self.cfg.BG_SCORE * bg, masks], 1)

            # note, that the masks contain the background as the first channel
            return logits_6, masks

        def forward(self, y, _=None, labels=None):
            test_mode = labels is None

            x = self.forward_backbone(y)

            cls = self.forward_cls(x)
            logits, masks = self.forward_mask(x, y.size()[-2:])

            if test_mode:
                return cls, masks

            # foreground stats
            b,c,h,w = masks.size()
            masks_ = masks.view(b,c,-1)
            masks_ = masks_[:, 1:]
            cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

            # upscale the masks & clean
            masks = self._rescale_and_clean(masks, y, labels)

            return cls, cls_fg, {"cam": masks}, logits, None, None, None

        def _rescale_and_clean(self, masks, image, labels):
            masks = F.interpolate(masks, size=image.size()[-2:], mode='bilinear', align_corners=True)
            masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
            return masks

    return CAM_MF