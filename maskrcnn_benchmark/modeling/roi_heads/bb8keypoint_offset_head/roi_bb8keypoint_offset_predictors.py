# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn


class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_bb8keypoints = config.MODEL.ROI_BB8KEYPOINT_OFFSET_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.bb8keypoint_offset_pred = nn.Linear(num_inputs, num_bb8keypoints * 2)

        nn.init.normal_(self.bb8keypoint_offset_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bb8keypoint_offset_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        bb8keypoint_offset_pred = self.bb8keypoint_offset_pred(x)
        return bb8keypoint_offset_pred


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BB8KEYPOINT_OFFSET_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.bb8keypoint_offset_pred = nn.Linear(representation_size, num_classes * 2)

        nn.init.normal_(self.bb8keypoint_offset_pred.weight, std=0.001)
        for l in [self.bb8keypoint_offset_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        bb8keypoint_deltas = self.bb8keypoint_offset_pred(x)

        return bb8keypoint_deltas


_ROI_BB8KEYPOINT_OFFSET_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
}


def make_roi_bb8keypoint_offset_predictor(cfg):
    func = _ROI_BB8KEYPOINT_OFFSET_PREDICTOR[cfg.MODEL.ROI_BB8KEYPOINT_OFFSET_HEAD.PREDICTOR]
    return func(cfg)
