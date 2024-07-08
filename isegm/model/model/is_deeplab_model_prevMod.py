import torch.nn as nn
import torch
from isegm.utils.serialization import serialize
from .is_model_prevMod import ISModel_prevMod, XConvBnRelu2
from .modeling.deeplab_v3 import DeepLabV3Plus
from .modeling.basic_blocks import SepConvHead
from isegm.model.modifiers import LRMult
from mmcv.cnn import ConvModule

class DeeplabModel_prevMod(ISModel_prevMod):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, aspp_dropout=0.5,
                 backbone_norm_layer=None, backbone_lr_mult=0.1, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch, project_dropout=aspp_dropout,
                                               norm_layer=norm_layer, backbone_norm_layer=backbone_norm_layer)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))
        self.head = SepConvHead(1, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
                                num_layers=2, norm_layer=norm_layer)

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
                out_channels=deeplab_ch,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            XConvBnRelu2(deeplab_ch, deeplab_ch),
        )

        self.refine_fusion = nn.Sequential(
            XConvBnRelu2(deeplab_ch * 2, deeplab_ch),
            XConvBnRelu2(deeplab_ch, deeplab_ch)
        )

    def backbone_forward(self, image, coord_features, prev_mask, prev_mask_modulated):
        pred_feature = self.feature_extractor(image, coord_features)[0]
        prevMod_feature = self.conv_prevMod(torch.cat((image, prev_mask, prev_mask_modulated), dim=1))

        fused_feature = self.refine_fusion(torch.cat([pred_feature, prevMod_feature], dim=1))
        pred_logits = self.head(fused_feature)

        return {'instances': pred_logits, 'instances_aux':None }