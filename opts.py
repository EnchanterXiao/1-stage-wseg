from __future__ import print_function

import os
import torch
import argparse
from core.config import cfg


def add_global_arguments(parser):

    parser.add_argument("--dataset", type=str,
                        help="Determines dataloader to use (only Pascal VOC supported)")
    parser.add_argument("--exp", type=str, default="main",
                        help="ID of the experiment (multiple runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Snapshot \"ID,iter\" to load")
    parser.add_argument("--run", type=str, help="ID of the run")
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument("--snapshot-dir", type=str, default='../1sw/snapshots',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--logdir", type=str, default='../1sw/logs',
                        help="Where to save log files of the model.")

    # used at inference only
    parser.add_argument("--infer-list", type=str, default='../1sw/data/val_augvoc.txt',
                        help="Path to a file list")
    parser.add_argument("--mask-output-dir", type=str, default='results/',
                        help="Path where to save masks")

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument("--random-seed", type=int, default=64, help="Random seed")


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_global_arguments(args):

    torch.set_num_threads(args.workers)
    if args.workers != torch.get_num_threads():
        print("Warning: # of threads is only ", torch.get_num_threads())

    setattr(args, "fixed_batch_path", os.path.join(args.logdir, args.dataset, args.exp, "fixed_batch.pt"))
    args.logdir = os.path.join(args.logdir, args.dataset, args.exp, args.run)
    maybe_create_dir(args.logdir)
    #print("Saving events in: {}".format(args.logdir))

    #
    # Model directories
    #
    args.snapshot_dir = os.path.join(args.snapshot_dir, args.dataset, args.exp, args.run)
    maybe_create_dir(args.snapshot_dir)
    #print("Saving snapshots in: {}".format(args.snapshot_dir))


def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")
    add_global_arguments(parser)
    args = parser.parse_args(args_in)
    check_global_arguments(args)

    return args

def get_deeplabarguments(args_in):
    parser = argparse.ArgumentParser(description="Model Evaluation")
    add_global_arguments(parser)
    parser.add_argument('--scoremap_path', type=str, default='',
                        help='backbone name (default: resnet)')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # parser.add_argument('--workers', type=int, default=4,
    #                     metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                        testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.007, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                                comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args(args_in)
    check_global_arguments(args)
    return args
