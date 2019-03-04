from torch import nn

from maskrcnn_benchmark import layers
from maskrcnn_benchmark.modeling import registry


@registry.ROI_BB8KEYPOINT_PREDICTOR.register("BB8KeypointRCNNPredictor")
class BB8KeypointRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(BB8KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        num_keypoints = cfg.MODEL.ROI_BB8KEYPOINT_HEAD.NUM_CLASSES
        deconv_kernel = 4
        self.kps_score_lowres = layers.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = layers.interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x


def make_roi_bb8keypoint_predictor(cfg, in_channels):
    func = registry.ROI_BB8KEYPOINT_PREDICTOR[cfg.MODEL.ROI_BB8KEYPOINT_HEAD.PREDICTOR]
    return func(cfg, in_channels)