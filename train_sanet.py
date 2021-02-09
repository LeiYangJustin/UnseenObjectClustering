"""Adopted from https://github.com/NVlabs/UnseenObjectClustering/blob/master/.gitignore"""

import torch
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import argparse
import pprint
import numpy as np
import sys
import os
import os.path as osp
import cv2

import yaml
from easydict import EasyDict as edict
import datetime

"""Set up paths for UCN"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add lib to PYTHONPATH
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

import datasets
from depthOIS import SlotNet
from depthOIS import Trainer


"""Set config file"""
def cfg_from_file(filename, cfg):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    # for k, v in yaml_cfg.items():
    #     cfg[k] = v
    # return cfg
    return yaml_cfg


""" get output directory """
def get_output_dir(cfg, imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', cfg.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net)



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train My network')
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=30, type=int)
    parser.add_argument('--startepoch', dest='startepoch',
                        help='the starting epoch',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver type',
                        default='adam', type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    cfg = edict()
    if args.cfg_file is not None:
        cfg = cfg_from_file(args.cfg_file, cfg)

    """constants
    """
    # For reproducibility
    cfg.RNG_SEED = 3
    # A small number that's used many times
    cfg.EPS = 1e-14
    # Root directory of project
    cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__)))
    # Place outputs under an experiments directory
    cfg.EXP_DIR = cfg.EXP_DIR + datetime.datetime.now().strftime('%m%d_%H%M%S')
    # Default GPU device id
    cfg.GPU_ID = 0
    # No intrinsics
    cfg.INTRINSICS = ()
    # prepare dataset
    cfg.MODE = 'TRAIN'
    cfg.TRAIN.ITERS = 0
    cfg.TRAIN.SYN_CROP = False  
    cfg.TRAIN.SYN_CROP_SIZE = 224
    cfg.PARALLEL = False
    cfg.epochs = args.epochs
    cfg.resume = args.pretrained


    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)


    ## Dataset
    dataset = datasets.TableTopObject(image_set='train', cfg=cfg)
    # worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    num_workers = 0
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, 
        shuffle=True, drop_last=True, num_workers=num_workers)
    print('Use dataset `{:s}` for training'.format(dataset.name))

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    output_dir = get_output_dir(cfg, dataset, None)
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cfg.EXP_DIR = output_dir
    print(cfg.EXP_DIR)

    ## use parameters to initialize the network instead of using config
    model = SlotNet(cfg)

    # prepare optimizer
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY},
                    {'params': model.weight_parameters(), 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY}]

    print(len(model.bias_parameters()))

    print('=> setting {} solver'.format(args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, cfg.OPTIMIZER.LEARNING_RATE,
                                     betas=(cfg.OPTIMIZER.MOMENTUM, cfg.OPTIMIZER.BETA))
    else:
        raise NotImplementedError

    trainer = Trainer(model, optimizer, dataloader, config=cfg)

    # main loop
    for epoch in range(args.startepoch, args.epochs):
        # train for one epoch
        trainer.train_epoch(epoch, input_mode=cfg.INPUT)
        trainer._save_checkpoint(epoch)
