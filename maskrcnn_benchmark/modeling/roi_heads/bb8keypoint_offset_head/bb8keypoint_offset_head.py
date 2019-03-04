# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_bb8keypoint_offset_feature_extractors import make_roi_bb8keypoint_offset_feature_extractor
from .roi_bb8keypoint_offset_predictors import make_roi_bb8keypoint_offset_predictor
from .inference import make_roi_bb8keypoint_offset_post_processor   # todo
from .loss import make_roi_bb8keypoint_offset_loss_evaluator   # todo

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIBB8KeypointHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBB8KeypointHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_bb8keypoint_offset_feature_extractor(cfg)
        self.predictor = make_roi_bb8keypoint_offset_predictor(cfg)
        self.post_processor = make_roi_bb8keypoint_offset_post_processor(cfg)
        self.loss_evaluator = make_roi_bb8keypoint_offset_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        if self.training and self.cfg.MODEL.ROI_BB8KEYPOINT_OFFSET_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
        else:
            x = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        bb8keypoint_delta_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor(bb8keypoint_delta_regression, proposals)   # todo
            return x, result, {}

        loss_bb8keypoint_offset_reg = self.loss_evaluator(
            bb8keypoint_delta_regression
        )
        return (
            x,
            proposals,
            dict(loss_bb8keypoint_offset_reg=loss_bb8keypoint_offset_reg),
        )


def build_roi_bb8keypoint_offset_head(cfg):
    """
    Constructs a new bb8keypoint head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBB8KeypointHead(cfg)
