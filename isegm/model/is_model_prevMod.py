import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, BatchImageNormalize, ScaleLayer
from isegm.model.modulation import modulate_prevMask
import math


class ISModel_prevMod(nn.Module):
    def __init__(self, with_aux_output=False, norm_radius=5, use_disks=False, cpu_dist_maps=False,
                 norm_layer=nn.BatchNorm2d,
                 use_rgb_conv=False, use_leaky_relu=False,  # the two arguments only used for RITM
                 with_prev_mask=False, norm_mean_std=([.485, .456, .406], [.229, .224, .225]), N=7, R_max=100):
        super().__init__()

        self.with_aux_output = with_aux_output
        self.with_prev_mask = with_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 4
        self.N = N
        self.R_max = R_max

        if use_rgb_conv:
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)
        else:
            self.maps_transform = nn.Identity()

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=cpu_dist_maps, use_disks=use_disks)

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)

        if torch.all(prev_mask == torch.zeros_like(prev_mask)):
            prev_masks = torch.cat([prev_mask, prev_mask], dim=1)
            coord_features = self.get_coord_features(image, prev_masks, points)
            coord_features = self.maps_transform(coord_features)
            outputs = self.backbone_forward(image, coord_features, prev_mask, prev_mask)
            outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:], mode='bilinear', align_corners=True)
            prev_mask = torch.sigmoid(outputs['instances'])
            
        prev_mask_modulated = modulate_prevMask(prev_mask, points, self.N, self.R_max)
        prev_masks = torch.cat([prev_mask, prev_mask_modulated], dim=1)

        coord_features = self.get_coord_features(image, prev_masks, points)
        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, coord_features, prev_mask, prev_mask_modulated)

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)

        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                                 mode='bilinear', align_corners=True)

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_masks, points):
        coord_features = self.dist_maps(image, points)
        if prev_masks is not None:
            coord_features = torch.cat((prev_masks, coord_features), dim=1)

        return coord_features

    def get_last_point(self, points):
        last_point = torch.zeros((points.shape[0], 1, 4), device=points.device, dtype=points.dtype)
        last_point[:, 0, :3] = points[points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1)]
        last_point[:, 0, -1][
            torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] < points.shape[
                1] // 2] = 1
        last_point[:, 0, -1][
            torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] >= points.shape[
                1] // 2] = 0

        return last_point



class XConvBnRelu2(nn.Module):
    """
    Xception conv bn relu
    """

    def __init__(self, input_dims=3, out_dims=16, **kwargs):
        super(XConvBnRelu2, self).__init__()
        self.conv3x3_1 = nn.Conv2d(input_dims, input_dims, 3, 1, 1, groups=input_dims)
        self.norm1 = nn.BatchNorm2d(input_dims)
        self.conv3x3_2 = nn.Conv2d(input_dims, input_dims, 3, 1, 1, groups=input_dims)
        self.conv1x1 = nn.Conv2d(input_dims, out_dims, 1, 1, 0)
        self.norm2 = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv3x3_1(x)
        x = self.norm1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
