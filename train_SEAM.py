from __future__ import print_function

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from datasets import get_dataloader, get_num_classes, get_class_names
from models import get_model

from base_trainer import BaseTrainer
from functools import partial

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.utils import Colorize
from losses import get_criterion, mask_loss_ce

from utils.timer import Timer
from utils.stat_manager import StatManager
from utils.dcrf import crf_inference

# specific to pytorch-v1 cuda-9.0
# see: https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
# and: https://github.com/pytorch/pytorch/issues/14456
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
DEBUG = False
global weight_attention
weight_attention = 20


def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x

class DecTrainer(BaseTrainer):

    def __init__(self, args, **kwargs):
        super(DecTrainer, self).__init__(args, **kwargs)

        self.scale_factor = 0.5
        # dataloader
        self.trainloader = get_dataloader(args, cfg, cfg.DATASET.FILENAME)
        self.valloader = get_dataloader(args, cfg, 'val_voc')
        self.denorm = self.trainloader.dataset.denorm

        self.nclass = get_num_classes(args)
        self.classNames = get_class_names(args)[:-1]
        assert self.nclass == len(self.classNames)

        self.classIndex = {}
        for i, cname in enumerate(self.classNames):
            self.classIndex[cname] = i

        # model
        self.enc = get_model(cfg.NET, num_classes=self.nclass)
        self.criterion_cls = get_criterion(cfg.NET.LOSS)
        print(self.enc)

        # optimizer using different LR
        enc_params = self.enc.parameter_groups(cfg.NET.LR, cfg.NET.WEIGHT_DECAY)
        self.optim_enc = self.get_optim(enc_params, cfg.NET)

        # checkpoint management
        self._define_checkpoint('enc', self.enc, self.optim_enc)
        self._load_checkpoint(args.resume)

        self.fixed_batch = None
        self.fixed_batch_path = args.fixed_batch_path
        if os.path.isfile(self.fixed_batch_path):
            print("Loading fixed batch from {}".format(self.fixed_batch_path))
            self.fixed_batch = torch.load(self.fixed_batch_path)

        # using cuda
        self.enc = nn.DataParallel(self.enc).cuda()
        self.criterion_cls = nn.DataParallel(self.criterion_cls).cuda()

    def step(self, epoch, image, gt_labels, train=False, visualise=False):

        PRETRAIN = epoch < (11 if DEBUG else cfg.TRAIN.PRETRAIN)
        PRETRAIN_er = epoch < (11 if DEBUG else cfg.TRAIN.PRETRAIN + 5)
        # PRETRAIN = False
        # PRETRAIN_er = False


        # denorm image
        image_raw = self.denorm(image.clone())

        # classification
        cls_out, cls_fg, masks, mask_logits, pseudo_gt, loss_mask, loss_at = self.enc(image, image_raw, gt_labels)
        # classification loss
        loss_cls = self.criterion_cls(cls_out, gt_labels).mean()
        if train:
            image2 = F.interpolate(image, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            image2_raw = F.interpolate(image_raw, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            cls_out2, cls_fg2, masks2, mask_logits2, pseudo_gt2, loss_mask2, loss_at2 = self.enc(image2, image2_raw, gt_labels)
            if not PRETRAIN_er:
                loss_cls += self.criterion_cls(cls_out2, gt_labels).mean()
            mask_logits = F.interpolate(mask_logits, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            loss_er = torch.mean(torch.abs(mask_logits.detach()-mask_logits2)) * 0.01

        # keep track of all losses for logging
        losses = {"loss_cls": loss_cls.item(),
                  "loss_fg": cls_fg.mean().item()}
        loss = loss_cls.clone()
        # attention loss
        if args.isattention:
            # loss_at = torch.mean(loss_at, dim=0) * weight_attention
            loss_at = loss_at.mean() * weight_attention
            losses.update({"loss_at": loss_at.item()})
            loss += loss_at.clone()

        if "dec" in masks:
            loss_mask = loss_mask.mean()



            if not PRETRAIN:
                loss += cfg.NET.MASK_LOSS_BCE * loss_mask
            if not PRETRAIN_er:
                if train:
                    loss_mask += loss_mask2.mean()
                    loss += loss_er

            assert not "pseudo" in masks
            masks["pseudo"] = pseudo_gt
            losses["loss_mask"] = loss_mask.item()
            if train:
                losses['loss_er'] = loss_er.item()

        losses["loss"] = loss.item()

        if train:
            self.optim_enc.zero_grad()
            loss.backward()
            self.optim_enc.step()

        for mask_key, mask_val in masks.items():
            masks[mask_key] = masks[mask_key].detach()

        mask_logits = mask_logits.detach()

        if visualise:
            self._visualise(epoch, image, masks, mask_logits, cls_out, gt_labels)

        # make sure to cut the return values from graph
        return losses, cls_out.detach(), masks, mask_logits

    def train_epoch(self, epoch):
        self.enc.train()

        stat = StatManager()
        stat.add_val("loss")
        stat.add_val("loss_cls")
        stat.add_val("loss_fg")
        stat.add_val("loss_bce")

        # adding stats for classes
        timer = Timer("New Epoch: ")
        train_step = partial(self.step, train=True, visualise=False)

        for i, (image, gt_labels, _, gt_masks, _) in enumerate(self.trainloader):

            # masks
            losses, _, _, _ = train_step(epoch, image, gt_labels)

            if self.fixed_batch is None:
                self.fixed_batch = {}
                self.fixed_batch["image"] = image.clone()
                self.fixed_batch["labels"] = gt_labels.clone()
                torch.save(self.fixed_batch, self.fixed_batch_path)

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            # intermediate logging
            if i % 10 == 0:
                msg = "Epoch[{}] Loss [{:04d}]: ".format(epoch, i)
                for loss_key, loss_val in losses.items():
                    msg += "{}: {:.4f} | ".format(loss_key, loss_val)

                msg += " | Im/Sec: {:.1f}".format(i * cfg.TRAIN.BATCH_SIZE / timer.get_stage_elapsed())
                print(msg)
                sys.stdout.flush()

            del image, gt_labels

            if DEBUG and i > 100:
                break

        def publish_loss(stats, name, t, prefix='data/'):
            print("{}: {:4.3f}".format(name, stats.summarize_key(name)))
            # self.writer.add_scalar(prefix + name, stats.summarize_key(name), t)

        for stat_key in stat.vals.keys():
            publish_loss(stat, stat_key, epoch)

        # plotting learning rate
        for ii, l in enumerate(self.optim_enc.param_groups):
            print("Learning rate [{}]: {:4.3e}".format(ii, l['lr']))
            self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

        # self.writer.add_scalar('lr/bg_baseline', self.enc.module.mean.item(), epoch)

        # visualising
        self.enc.eval()
        with torch.no_grad():
            self.step(epoch, self.fixed_batch["image"], \
                      self.fixed_batch["labels"], \
                      train=False, visualise=True)

    def _mask_rgb(self, masks, image_norm):
        # visualising masks
        masks_conf, masks_idx = torch.max(masks, 1)
        masks_conf = masks_conf - F.relu(masks_conf - 1, 0)

        masks_idx_rgb = self._apply_cmap(masks_idx.cpu(), masks_conf.cpu())
        return 0.3 * image_norm + 0.7 * masks_idx_rgb

    def _init_norm(self):
        self.trainloader.dataset.set_norm(self.enc.normalize)
        self.valloader.dataset.set_norm(self.enc.normalize)

    def _apply_cmap(self, mask_idx, mask_conf):
        palette = self.trainloader.dataset.get_palette()

        masks = []
        col = Colorize()
        mask_conf = mask_conf.float() / 255.0
        for mask, conf in zip(mask_idx.split(1), mask_conf.split(1)):
            m = col(mask).float()
            m = m * conf
            masks.append(m[None, ...])

        return torch.cat(masks, 0)

    def validation(self, epoch, writer, loader, checkpoint=False):

        stat = StatManager()

        # Fast test during the training
        def eval_batch(image, gt_labels):

            losses, cls, masks, mask_logits = \
                self.step(epoch, image, gt_labels, train=False, visualise=False)

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            return cls.cpu(), masks, mask_logits.cpu()

        self.enc.eval()

        # class ground truth
        targets_all = []

        # class predictions
        preds_all = []

        def add_stats(means, stds, x):
            means.append(x.mean())
            stds.append(x.std())

        for n, (image, gt_labels, _, gt_masks, _) in enumerate(loader):
            with torch.no_grad():
                cls_raw, masks_all, mask_logits = eval_batch(image, gt_labels)

            cls_sigmoid = torch.sigmoid(cls_raw).numpy()

            preds_all.append(cls_sigmoid)
            targets_all.append(gt_labels.cpu().numpy())

        #
        # classification
        #
        targets_stacked = np.vstack(targets_all)
        preds_stacked = np.vstack(preds_all)
        aps = average_precision_score(targets_stacked, preds_stacked, average=None)

        # offset = 0
        # skip BG AP
        offset = self.nclass - aps.size
        assert offset == 1, 'Class number mismatch'

        classNames = self.classNames[offset:]
        for ni, className in enumerate(classNames):
            writer.add_scalar('%02d_%s/AP' % (ni + offset, className), aps[ni], epoch)
            print("AP_{}: {:4.3f}".format(className, aps[ni]))

        meanAP = np.mean(aps)
        writer.add_scalar('all_wo_BG/mAP', meanAP, epoch)
        print('mAP: {:4.3f}'.format(meanAP))

        # total classification loss
        for stat_key in stat.vals.keys():
            writer.add_scalar('all/{}'.format(stat_key), stat.summarize_key(stat_key), epoch)

        if checkpoint and epoch >= cfg.TRAIN.PRETRAIN:
            # we will use mAP - mask_loss as our proxy score
            # to save the best checkpoint so far
            proxy_score = 1 - stat.summarize_key("loss")
            writer.add_scalar('all/checkpoint_score', proxy_score, epoch)
            self.checkpoint_best(proxy_score, epoch)

    def _visualise(self, epoch, image, masks, mask_logits, cls_out, gt_labels):
        image_norm = self.denorm(image.clone()).cpu()
        visual = [image_norm]

        if "cam" in masks:
            visual.append(self._mask_rgb(masks["cam"], image_norm))

        if "dec" in masks:
            visual.append(self._mask_rgb(masks["dec"], image_norm))

        if "pseudo" in masks:
            pseudo_gt_rgb = self._mask_rgb(masks["pseudo"], image_norm)

            # cancel ambiguous
            ambiguous = 1 - masks["pseudo"].sum(1, keepdim=True).cpu()
            pseudo_gt_rgb = ambiguous * image_norm + (1 - ambiguous) * pseudo_gt_rgb
            visual.append(pseudo_gt_rgb)

        # ready to assemble
        visual_logits = torch.cat(visual, -1)
        self._visualise_grid(visual_logits, gt_labels, epoch, scores=cls_out)


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Config: \n", cfg)

    trainer = DecTrainer(args)
    torch.manual_seed(0)

    timer = Timer()


    def time_call(func, msg, *args, **kwargs):
        timer.reset_stage()
        func(*args, **kwargs)
        print(msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))


    for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        print("Epoch >>> ", epoch)

        with torch.no_grad():
            if epoch == 0:
                pass
            else:
                time_call(trainer.validation, "Validation /   Val: ", epoch, trainer.writer_val, trainer.valloader,
                          checkpoint=True)
        time_call(trainer.train_epoch, "Train epoch: ", epoch)
