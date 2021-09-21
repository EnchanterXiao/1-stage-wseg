"""
Evaluating class activation maps from a given snapshot
Supports multi-scale fusion of the masks
Based on PSA
"""

import os
import sys
import numpy as np
import scipy
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torch.backends import cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

from opts import get_arguments, get_deeplabarguments
from core.config import cfg, cfg_from_file, cfg_from_list
from models.deeplabV3 import DeepLab

from utils.checkpoints import Checkpoint
from utils.timer import Timer
from utils.dcrf import crf_inference
from utils.inference_tools import get_inference_io
from datasets import get_dataloader, get_num_classes, get_class_names

def check_dir(base_path, name):

    # create the directory
    fullpath = os.path.join(base_path, name)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    return fullpath

def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def undo(self, imgarr):
        proc_img = imgarr.copy()

        proc_img[..., 0] = (self.std[0] * imgarr[..., 0] + self.mean[0]) * 255.
        proc_img[..., 1] = (self.std[1] * imgarr[..., 1] + self.mean[1]) * 255.
        proc_img[..., 2] = (self.std[2] * imgarr[..., 2] + self.mean[2]) * 255.

        return proc_img

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


if __name__ == '__main__':


    # loading the model
    # args = get_arguments(sys.argv[1:])
    args = get_deeplabarguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # initialising the dirs
    check_dir(args.mask_output_dir, "vis")
    check_dir(args.mask_output_dir, "no_crf")

    # Loading the model
    nclass = get_num_classes(args)
    model = DeepLab(num_classes=nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
    checkpoint = Checkpoint(args.snapshot_dir, max_n=5)
    checkpoint.add_model('enc', model)
    checkpoint.load(args.resume)
    print('load from', args.resume)

    for p in model.parameters():
        p.requires_grad = False

    # setting the evaluation mode
    model.eval()

    # assert hasattr(model, 'normalize')
    transform = tf.Compose([np.asarray,  Normalize()])

    WriterClass, DatasetClass = get_inference_io(cfg.TEST.METHOD)

    dataset = DatasetClass(args.infer_list, cfg.TEST, transform=transform)

    dataloader = DataLoader(dataset, shuffle=False, num_workers=args.workers,
                            pin_memory=True, batch_size=cfg.TEST.BATCH_SIZE)

    model = nn.DataParallel(model).cuda()

    timer = Timer()
    N = len(dataloader)

    palette = dataset.get_palette()
    pool = mp.Pool(processes=args.workers)
    writer = WriterClass(cfg.TEST, palette, args.mask_output_dir,
                         prospect_thresh=0,
                         heatmap=False, scoremap=False, CRF=False)

    for iter, (img_name, img_orig, images_in, pads, labels, gt_mask) in enumerate(tqdm(dataloader)):

        # cutting the masks
        masks = []

        with torch.no_grad():
            masks_pred = model(images_in)

        # saving the raw npy
        image = dataset.denorm(img_orig[0]).numpy()
        masks_pred = masks_pred.cpu()
        labels = torch.ones_like(labels[0])
        # labels = labels.type_as(masks_pred)
        # print(masks_pred.size())
        # print(labels.size())
        writer.save(img_name[0], image, masks_pred, pads, labels, gt_mask[0])
        # pool.apply_async(writer.save, args=(img_name[0], image, masks_pred, pads, labels, gt_mask[0]))

        timer.update_progress(float(iter + 1) / N)
        if iter % 100 == 0:
            msg = "Finish time: {}".format(timer.str_est_finish())
            tqdm.write(msg)
            sys.stdout.flush()

    pool.close()
    pool.join()