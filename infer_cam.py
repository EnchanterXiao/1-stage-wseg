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
from utils.timer import Timer
from utils.inference_tools import get_inference_io,MergeSingleScale


from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
import sys
from core.config import cfg, cfg_from_file, cfg_from_list
from models import get_model

from utils.checkpoints import Checkpoint
from opts import get_cam_arguments
from datasets.pascal_voc_ms import SinglescaleLoader

def check_dir(base_path, name):

    # create the directory
    fullpath = os.path.join(base_path, name)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    return fullpath

def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

if __name__ == '__main__':
    # prospect_threshs = [0.5]
    # heatmaps = [False]
    # scoremaps = [False]
    # CRFs = [False]

    test_id = [3]
    prospect_threshs = [0.0, 0.1, 0.3, 0.5, 0.7]
    heatmaps = [False, False, False, False, False, False]
    scoremaps = [False, False, False, False, False, False]
    CRFs = [False, False, False, False, False, False]

    # loading the model
    # args = get_arguments(sys.argv[1:])
    args = get_cam_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    # initialising the dirs
    for prospect_thresh, heatmap, scoremap in zip(prospect_threshs, heatmaps, scoremaps):
        check_dir(args.mask_output_dir+'_'+str(prospect_thresh).split('.')[-1], "vis")
        check_dir(args.mask_output_dir+'_'+str(prospect_thresh).split('.')[-1], "crf")
        check_dir(args.mask_output_dir+'_'+str(prospect_thresh).split('.')[-1], "no_crf")
        if heatmap:
            check_dir(args.mask_output_dir+'_'+str(prospect_thresh).split('.')[-1], "heatmap")
        if scoremap:
            check_dir(args.mask_output_dir+'_'+str(prospect_thresh).split('.')[-1], "scoremap")

    # Loading the model
    model = get_model(cfg.NET, num_classes=cfg.TEST.NUM_CLASSES)
    checkpoint = Checkpoint(args.snapshot_dir, max_n=5)
    checkpoint.add_model('enc', model)
    checkpoint.load(args.resume)
    target_layers = [model.cls_branch[-1]]

    assert hasattr(model, 'normalize')
    transform = tf.Compose([np.asarray, model.normalize])

    # WriterClass, _ = get_inference_io(cfg.TEST.METHOD)
    WriterClass = MergeSingleScale
    DatasetClass = SinglescaleLoader

    dataset = DatasetClass(args.infer_list, cfg.TEST, transform=transform)

    dataloader = DataLoader(dataset, shuffle=False, num_workers=args.workers, \
                            pin_memory=True, batch_size=cfg.TEST.BATCH_SIZE)

    # model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    timer = Timer()
    N = len(dataloader)

    palette = dataset.get_palette()
    writers = []

    ###grad_cam
    cam_algorithm = methods[args.method]
    # cam = cam_algorithm(model=model,
    #                    target_layers=target_layers,
    #                    use_cuda=args.use_cuda)
    # cam.batch_size = 32

    cam = cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda)
    cam.batch_size = 32
    for idx, (prospect_thresh, heatmap, scoremap, crf) in enumerate(zip(prospect_threshs, heatmaps, scoremaps, CRFs)):
        writers.append(WriterClass(cfg.TEST, palette, args.mask_output_dir + '_' + str(prospect_thresh).split('.')[-1],
                             prospect_thresh=prospect_thresh,
                             heatmap=heatmap, scoremap=scoremap, CRF=crf))
        if idx not in test_id:
            continue
        pool = mp.Pool(processes=args.workers)

        for iter, (img_name, img_orig, images_in, labels, gt_mask) in enumerate(tqdm(dataloader)):


            images_in = images_in.permute(0, 3, 1, 2)
            bs, c, h, w = images_in.size()
            masks_pred = np.zeros((1, cfg.TEST.NUM_CLASSES, h, w))

            new_labels = np.where(labels[0].numpy()==1)[0]
            for label in new_labels:
                masks_pred[0, label+1, :, :] = cam(input_tensor=images_in,
                                    target_category=[label],
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)[0, :]

            # saving the raw npy
            labels = labels[0]
            image = dataset.denorm(img_orig[0]).numpy()
            masks_pred = torch.from_numpy(masks_pred) #.cpu()
            labels = labels.type_as(masks_pred)
            # pool.apply_async(writers[idx].save, args=(img_name[0], image, masks_pred, labels, gt_mask[0]))
            writers[idx].save(img_name[0], image, masks_pred, labels, gt_mask[0])
            timer.update_progress(float(iter + 1) / N)
            if iter % 100 == 0:
                msg = "Finish time: {}".format(timer.str_est_finish())
                tqdm.write(msg)
                sys.stdout.flush()
        pool.close()
        pool.join()

