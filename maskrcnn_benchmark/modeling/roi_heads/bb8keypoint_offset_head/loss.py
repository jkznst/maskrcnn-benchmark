import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss

def bb8keypoint_offset_encode(bb8keypoints, reference_proposals):
    '''
    :param bb8keypoints: shape (N, 8, 3)
    :param reference_proposals: (N, 4)
    :return:
    '''
    TO_REMOVE = 1  # TODO remove
    ex_widths = reference_proposals[:, 2:3] - reference_proposals[:, 0:1] + TO_REMOVE
    ex_heights = reference_proposals[:, 3:4] - reference_proposals[:, 1:2] + TO_REMOVE
    ex_ctr_x = reference_proposals[:, 0:1] + 0.5 * ex_widths
    ex_ctr_y = reference_proposals[:, 1:2] + 0.5 * ex_heights

    bb8keypoints = bb8keypoints.view(bb8keypoints.shape[0], -1, 3)
    gt_bb8keypoint_x = bb8keypoints[:, :, 0]
    gt_bb8keypoint_y = bb8keypoints[:, :, 1]
    # currently ignore visibility

    wx, wy = 10., 10.
    targets_dx = wx * (gt_bb8keypoint_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_bb8keypoint_y - ex_ctr_y) / ex_heights

    targets = torch.stack((targets_dx, targets_dy), dim=-1)
    targets = targets.view(targets.shape[0], -1)    # shape (N, 16) xyxy

    return targets.to(reference_proposals.device, dtype=torch.float32)

class BB8KeypointOffsetRCNNLossComputation(object):

    def __init__(self, proposal_matcher):
        """
        Arguments:
            proposal_matcher (Matcher)
        """
        self.proposal_matcher = proposal_matcher

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Keypoint RCNN needs "labels" and "keypoints "fields for creating the targets
        target = target.copy_with_fields(["labels", "bb8keypoints"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        bb8keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            # matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = proposals_per_image.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # bb8keypoint scores are only computed on positive samples
            # positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            bb8kp = matched_targets.get_field("bb8keypoints")
            # bb8keypoints = bb8keypoints[positive_inds]

            # positive_proposals = proposals_per_image[positive_inds]

            # compute bb8keypoint offset regression targets
            # regression_targets_per_image = bb8keypoint_offset_encode(
            #     bb8keypoints, positive_proposals.bbox
            # )

            labels.append(labels_per_image)
            bb8keypoints.append(bb8kp)

        return labels, bb8keypoints

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, keypoints = self.prepare_targets(proposals, targets)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, keypoints_per_image, proposals_per_image in zip(
            labels, keypoints, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "bb8keypoints", keypoints_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        # for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
        #     zip(sampled_pos_inds, sampled_neg_inds)
        # ):
        #     # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
        #     img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
        #     proposals_per_image = proposals[img_idx][img_sampled_inds]
        #     proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, keypoint_offset_pred):
        '''
        :param proposals: (list[BoxList])
        :param keypoint_offset_pred:
        :return:
        '''
        bb8_keypoint_offset_targets = []
        positive_inds = []
        for proposals_per_image in self._proposals:
            bb8kp = proposals_per_image.get_field("bb8keypoints")
            labels_per_image = proposals_per_image.get_field("labels")

            positive_inds_per_image = torch.nonzero(labels_per_image > 0).squeeze(1)

            bb8kp = bb8kp[positive_inds_per_image]
            positive_proposals = proposals_per_image[positive_inds_per_image]

            # compute bb8keypoint offset regression targets
            regression_targets_per_image = bb8keypoint_offset_encode(
                bb8kp.keypoints, positive_proposals.bbox)
            bb8_keypoint_offset_targets.append(regression_targets_per_image)
            positive_inds.append(positive_inds_per_image)

        bb8_keypoint_offset_targets = cat(bb8_keypoint_offset_targets, dim=0)
        positive_inds = cat(positive_inds, dim=0)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing, for class-specific regression
        # sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        # labels_pos = labels[sampled_pos_inds_subset]
        # map_inds = 16 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8,
        #                                                     9, 10, 11, 12, 13, 14, 15], device=device)

        if bb8_keypoint_offset_targets.numel() == 0:
            return bb8_keypoint_offset_targets.sum() * 0

        # print("keypoint_offset_pred.device:{}".format(keypoint_offset_pred.device))
        # print("keypoint_offset_target.device:{}".format(bb8_keypoint_offset_targets.device))
        keypoint_loss = smooth_l1_loss(
            keypoint_offset_pred[positive_inds],
            bb8_keypoint_offset_targets,
            size_average=False,
            beta=1,
        )
        keypoint_loss = keypoint_loss / keypoint_offset_pred.shape[0]
        return keypoint_loss


def make_roi_bb8keypoint_offset_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = BB8KeypointOffsetRCNNLossComputation(matcher)
    return loss_evaluator
