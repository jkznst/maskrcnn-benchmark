#!/usr/bin/env bash

source activate fb_pytorch_maskrcnn
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ./configs/e2e_mask_rcnn_R_50_FPN_1x_2gpu.yaml SOLVER.IMS_PER_BATCH 4
