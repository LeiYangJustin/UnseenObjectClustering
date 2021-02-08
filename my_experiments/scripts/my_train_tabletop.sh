#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./my_tools/train_net.py \
  --network seg_resnet34_8s_embedding \
  --dataset tabletop_object_train \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
  --solver adam \
  --epochs 16