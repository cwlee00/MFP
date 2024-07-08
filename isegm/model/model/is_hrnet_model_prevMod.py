import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from isegm.utils.serialization import serialize
from .is_model_prevMod import ISModel_prevMod, XConvBnRelu2
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult


class HRNetModel_prevMod(ISModel_prevMod):
    @serialize
    def __init__(self, width=48, ocr_width=256, small=False, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = HighResolutionNet(width=width, ocr_width=ocr_width, small=small,
                                                   num_classes=1, norm_layer=norm_layer)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        if ocr_width > 0:
            self.feature_extractor.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.conv3x3_ocr.apply(LRMult(1.0))

        self.conv_prevMod = nn.Sequential(
            ConvModule(
                in_channels=5,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvModule(
                in_channels=64,
                out_channels=270,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            XConvBnRelu2(270, 270),
        )

        self.refine_fusion = nn.Sequential(
            XConvBnRelu2(270 * 2, 270),
            XConvBnRelu2(270, 270)
        )

    def backbone_forward(self, image, coord_features, prev_mask, prev_mask_modulated):
        pred_feature = self.feature_extractor.compute_hrnet_feats(image, coord_features)
        prevMod_feature = self.conv_prevMod(torch.cat((image, prev_mask, prev_mask_modulated), dim=1))

        fused_feature = self.refine_fusion(torch.cat([pred_feature, prevMod_feature], dim=1))

        if self.feature_extractor.ocr_width > 0:
            out_aux = self.feature_extractor.aux_head(fused_feature)
            feats = self.feature_extractor.conv3x3_ocr(fused_feature)

            context = self.feature_extractor.ocr_gather_head(feats, out_aux)
            feats = self.feature_extractor.ocr_distri_head(feats, context)
            pred_logits = self.feature_extractor.cls_head(feats)
        else:
            pred_logits = self.feature_extractor.cls_head(fused_feature)

        return {'instances': pred_logits, 'instances_aux': out_aux}


