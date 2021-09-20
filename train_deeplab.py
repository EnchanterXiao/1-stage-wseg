from __future__ import print_function

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


from datasets import get_dataloader, get_num_classes, get_class_names
from models.deeplabV3 import DeepLab

from base_trainer import BaseTrainer
from functools import partial

from opts import get_arguments, get_deeplabarguments
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.utils import Colorize
from losses import get_criterion, mask_loss_ce, SegmentationLosses

from utils.timer import Timer
from utils.stat_manager import StatManager
from utils.calculate_weights import calculate_weigths_labels
from utils.metrics import Metric
from datasets.pascal_voc import PascalVOC

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x


class DecTrainer(BaseTrainer):

    def __init__(self, args, **kwargs):
        super(DecTrainer, self).__init__(args, **kwargs)

        # dataloader
        self.trainloader = get_dataloader(args, cfg, cfg.DATASET.FILENAME, cfg.DATASET.SCOREMAP)
        self.valloader = get_dataloader(args, cfg, 'val_voc')
        self.denorm = self.trainloader.dataset.denorm

        self.nclass = get_num_classes(args)
        self.classNames = get_class_names(args)[:-1]
        assert self.nclass == len(self.classNames)

        self.classIndex = {}
        for i, cname in enumerate(self.classNames):
            self.classIndex[cname] = i

        # model
        # Define network
        self.enc = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': self.enc.get_1x_lr_params(), 'lr': args.lr},
                        {'params': self.enc.get_10x_lr_params(), 'lr': args.lr * 10}]
        print(self.enc)

        # optimizer using different LR
        # Define Optimizer
        self.optim_enc = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(cfg.DATASET.ROOT, cfg.DATASET.NAME, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=True).build_loss(mode=args.loss_type)

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
        self.criterion_cls = nn.DataParallel(self.criterion).cuda()

    def step(self, image, gt_labels, train=False):

        # denorm image
        image_raw = self.denorm(image.clone())
        cls_labels = gt_labels[0]
        mask_label = gt_labels[1]
        score_label = gt_labels[2]

        output = self.enc(image_raw)

        #seg_loss
        loss = self.criterion(output, mask_label)

        # classification loss
        # loss_cls = self.criterion_cls(cls_out, gt_labels).mean()

        # keep track of all losses for logging
        losses = {}
        losses["loss"] = loss.item()

        if train:
            self.optim_enc.zero_grad()
            loss.backward()
            self.optim_enc.step()

        mask_logits = output.detach()

        # make sure to cut the return values from graph
        return losses, output, mask_logits

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

        for i, (image, gt_labels, _, gt_masks) in enumerate(self.trainloader):

            # masks
            losses, _, _ = train_step(image, [gt_labels, gt_masks, _])

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

    def validation(self, epoch, writer, loader, checkpoint=False):

        stat = StatManager()

        # Fast test during the training
        def eval_batch(image, gt_labels):

            losses, masks, mask_logits = \
                self.step(image, gt_labels, train=False, visualise=False)

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            return masks, mask_logits.cpu()

        self.enc.eval()

        conf_mat = np.zeros((21, 21))
        class_stats = {}
        for class_idx in range(21):
            class_stats[class_idx] = []

        # count of the images
        num_im = 0

        for n, (image, gt_labels, _, gt_masks) in enumerate(loader):
            with torch.no_grad():
                losses, masks_all, mask_logits = eval_batch(image, [gt_labels, gt_masks, _])

                evaluate_one(conf_mat, gt_masks, mask_logits)
                num_im += 1

        mIOU = summarise_stats(conf_mat)
        # total classification loss
        writer.add_scalar('mIOU', mIOU, epoch)

        if checkpoint and epoch >= cfg.TRAIN.PRETRAIN:
            # we will use mAP - mask_loss as our proxy score
            # to save the best checkpoint so far
            proxy_score = mIOU
            writer.add_scalar('all/checkpoint_score', proxy_score, epoch)
            self.checkpoint_best(proxy_score, epoch)

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


def evaluate_one(conf_mat, mask_gt, mask):

    gt = mask_gt.reshape(-1)
    pred = mask.reshape(-1)

    assert(len(gt) == len(pred))

    for i in range(len(gt)):
        if gt[i] < conf_mat.shape[0]:
            conf_mat[gt[i], pred[i]] += 1.0
    return


def get_stats(M, i):

    TP = M[i, i]
    FN = np.sum(M[i, :]) - TP # false negatives
    FP = np.sum(M[:, i]) - TP # false positives

    return TP, FN, FP


def summarise_stats(M):

    eps = 1e-20

    mean = Metric()
    mean.add_metric(Metric.IoU)
    mean.add_metric(Metric.Precision)
    mean.add_metric(Metric.Recall)

    mean_bkg = Metric()
    mean_bkg.add_metric(Metric.IoU)
    mean_bkg.add_metric(Metric.Precision)
    mean_bkg.add_metric(Metric.Recall)

    head_fmt = "{:>12} | {:>5}" + " | {:>5}"*3
    row_fmt = "{:>12} | {:>5}" + " | {:>5.1f}"*3
    split = "-"*44

    def print_row(fmt, row):
        print(fmt.format(*row))

    print_row(head_fmt, ("Class", "#", "IoU", "Pr", "Re"))
    print(split)

    for cat in PascalVOC.CLASSES:

        i = PascalVOC.CLASS_IDX[cat]

        TP, FN, FP = get_stats(M, i)

        iou = 100. * TP / (eps + FN + FP + TP)
        pr = 100. * TP / (eps + TP + FP)
        re = 100. * TP / (eps + TP + FN)

        mean_bkg.update_value(Metric.IoU, iou)
        mean_bkg.update_value(Metric.Precision, pr)
        mean_bkg.update_value(Metric.Recall, re)

        if cat != "background":
            mean.update_value(Metric.IoU, iou)
            mean.update_value(Metric.Precision, pr)
            mean.update_value(Metric.Recall, re)

        count = int(np.sum(M[i, :]))
        print_row(row_fmt, (cat, count, iou, pr, re))


    print(split)
    sys.stdout.write("mIoU: {:.2f}\t".format(mean.summarize(Metric.IoU)))
    sys.stdout.write("  Pr: {:.2f}\t".format(mean.summarize(Metric.Precision)))
    sys.stdout.write("  Re: {:.2f}\n".format(mean.summarize(Metric.Recall)))

    print(split)
    print("With background: ")
    sys.stdout.write("mIoU: {:.2f}\t".format(mean_bkg.summarize(Metric.IoU)))
    sys.stdout.write("  Pr: {:.2f}\t".format(mean_bkg.summarize(Metric.Precision)))
    sys.stdout.write("  Re: {:.2f}\n".format(mean_bkg.summarize(Metric.Recall)))
    return mean_bkg.summarize(Metric.IoU)


if __name__ == "__main__":
    args = get_deeplabarguments(sys.argv[1:])

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
                time_call(trainer.validation, "Validation /   Val: ", epoch, trainer.writer_val, trainer.valloader,
                          checkpoint=False)
            else:
                time_call(trainer.validation, "Validation /   Val: ", epoch, trainer.writer_val, trainer.valloader,
                          checkpoint=True)
        time_call(trainer.train_epoch, "Train epoch: ", epoch)
