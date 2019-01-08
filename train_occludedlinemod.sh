#!/usr/bin/env bash

source activate fb_pytorch_maskrcnn
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ./configs/occluded_linemod/e2e_faster_rcnn_R_50_C4_1x_2gpu_occluded.yaml
