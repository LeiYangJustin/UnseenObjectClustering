#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Train a UCN on image segmentation database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
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

"""Set up paths for UCN"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add lib to PYTHONPATH
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)


import datasets
import networks
# from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.train import train_segnet
from datasets.factory import get_dataset, get_tableobj_dataset



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
                        default='sgd', type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--dataset_background', dest='dataset_background_name',
                        help='background dataset to train on',
                        default='background_nvidia', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)

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
    cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    # Place outputs under an experiments directory
    cfg.EXP_DIR = 'default'
    # Default GPU device id
    cfg.GPU_ID = 0
    # No intrinsics
    cfg.INTRINSICS = ()
    # prepare dataset
    cfg.MODE = 'TRAIN'
    cfg.TRAIN.ITERS = 0
    cfg.TRAIN.SYN_CROP = False  
    cfg.TRAIN.SYN_CROP_SIZE = 224


    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # dataset = get_dataset(args.dataset_name)
    # dataset = get_tableobj_dataset(cfg.MODE.lower())
    dataset = datasets.TableTopObject(image_set='train', cfg=cfg)
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    num_workers = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True, 
        num_workers=num_workers, worker_init_fn=worker_init_fn)
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

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        if isinstance(network_data, dict) and 'model' in network_data:
            network_data = network_data['model']
        print("=> using pre-trained network '{}'".format(args.network_name))
    else:
        network_data = None
        print("=> creating network '{}'".format(args.network_name))

    ## use parameters to initialize the network instead of using config
    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data, cfg).cuda()
    if torch.cuda.device_count() > 1:
        cfg.TRAIN.GPUNUM = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    network = torch.nn.DataParallel(network).cuda()
    cudnn.benchmark = True

    # prepare optimizer
    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': network.module.bias_parameters(), 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY},
                    {'params': network.module.weight_parameters(), 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY}]

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, cfg.OPTIMIZER.LEARNING_RATE,
                                     betas=(cfg.OPTIMIZER.MOMENTUM, cfg.OPTIMIZER.BETA))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, cfg.OPTIMIZER.LEARNING_RATE,
                                    momentum=cfg.OPTIMIZER.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m - args.startepoch for m in cfg.OPTIMIZER.MILESTONES], gamma=cfg.OPTIMIZER.GAMMA)
    cfg.epochs = args.epochs

    # main loop
    for epoch in range(args.startepoch, args.epochs):
        if args.solver == 'sgd':
            scheduler.step()

        # train for one epoch
        train_segnet(dataloader, network, optimizer, epoch, cfg)

        # save checkpoint
        if (epoch+1) % cfg.TRAIN.SNAPSHOT_EPOCHS == 0 or epoch == args.epochs - 1:
            state = network.module.state_dict()
            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                     if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_epoch_{:d}'.format(epoch+1) + '.checkpoint.pth')
            torch.save(state, os.path.join(output_dir, filename))
            print(filename)

        ## no evaluation code?
