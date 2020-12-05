#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained output/tabletop_object/tabletop_object_train/seg_resnet34_8s_embedding_cosine_rgbd_cat_epoch_$2.checkpoint.pth  \
  --dataset tabletop_object_test \
  --dataset_background background_texture \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_cat_tabletop.yml
