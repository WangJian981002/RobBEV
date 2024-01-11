# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule,build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

@NECKS.register_module()
class FuseLayer(BaseModule):

    def __init__(self,
                 img_feat_channels,
                 pts_feat_channels,
                 out_channels,
                 conv_cfg=dict(type='Conv2d', bias=False),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super(FuseLayer, self).__init__(init_cfg)
        layer = [build_conv_layer(
                    conv_cfg,
                    img_feat_channels+pts_feat_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),]
        self.fuseConv = nn.Sequential(*layer)


    def forward(self, inputs):
        fused_feat = self.fuseConv(torch.cat(inputs,axis=1))
        return fused_feat

@NECKS.register_module()
class FuseLayerv2(BaseModule):
    #通道注意力
    def __init__(self,
                 img_feat_channels,
                 pts_feat_channels,
                 out_channels,
                 conv_cfg=dict(type='Conv2d', bias=False),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super(FuseLayerv2, self).__init__(init_cfg)
        layer = [build_conv_layer(
                    conv_cfg,
                    img_feat_channels+pts_feat_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),]
        self.fuseConv = nn.Sequential(*layer)

        self.lateral = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels, out_channels)
        self.act = nn.Sigmoid()



    def forward(self, inputs):
        fused_feat = self.fuseConv(torch.cat(inputs,axis=1))
        atten = self.act(self.linear(self.lateral(fused_feat).squeeze())).unsqueeze(-1).unsqueeze(-1) #(B,n_c,1,1)
        return fused_feat*atten

@NECKS.register_module()
class FuseLayerv3(BaseModule):
    #通道+空间（conv1×1）注意力
    def __init__(self,
                 img_feat_channels,
                 pts_feat_channels,
                 out_channels,
                 conv_cfg=dict(type='Conv2d', bias=False),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super(FuseLayerv3, self).__init__(init_cfg)
        layer = [build_conv_layer(
                    conv_cfg,
                    img_feat_channels+pts_feat_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),]
        self.fuseConv = nn.Sequential(*layer)

        self.lateral = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels, out_channels)
        self.act = nn.Sigmoid()

        self.spatial_conv = build_conv_layer(conv_cfg, out_channels, 1, 1, stride=1, padding=0)


    def forward(self, inputs):
        fused_feat = self.fuseConv(torch.cat(inputs,axis=1)) #(B,C,H,W)
        atten_channel = self.act(self.linear(self.lateral(fused_feat).squeeze())).unsqueeze(-1).unsqueeze(-1) #(B,C,1,1)
        atten_spatial = self.act(self.spatial_conv(fused_feat)) #(B,1,H,W)
        return fused_feat*atten_channel*atten_spatial

@NECKS.register_module()
class FuseLayerv4(BaseModule):
    # 通道+空间（conv3×3）注意力
    def __init__(self,
                 img_feat_channels,
                 pts_feat_channels,
                 out_channels,
                 conv_cfg=dict(type='Conv2d', bias=False),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super(FuseLayerv4, self).__init__(init_cfg)
        layer = [build_conv_layer(
                    conv_cfg,
                    img_feat_channels+pts_feat_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),]
        self.fuseConv = nn.Sequential(*layer)

        self.lateral = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels, out_channels)
        self.act = nn.Sigmoid()

        self.spatial_conv = build_conv_layer(conv_cfg, out_channels, 1, 3, stride=1, padding=1)


    def forward(self, inputs):
        fused_feat = self.fuseConv(torch.cat(inputs,axis=1)) #(B,C,H,W)
        atten_channel = self.act(self.linear(self.lateral(fused_feat).squeeze())).unsqueeze(-1).unsqueeze(-1) #(B,C,1,1)
        atten_spatial = self.act(self.spatial_conv(fused_feat)) #(B,1,H,W)
        return fused_feat*atten_channel*atten_spatial


####### useful utils
def get_reference_points(spatial_shapes,  # 多尺度feature map对应的h,w，shape为[num_level,2]
                         valid_ratios,  # 多尺度feature map对应的mask中有效的宽高比，shape为[B, num_levels, 2]
                         device='cpu'):
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        # 对于每一层feature map初始化每个参考点中心横纵坐标，加减0.5是确保每个初始点是在每个pixel的中心，例如[0.5,1.5,2.5, ...]
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=torch.float32, device=device))

        # 将横纵坐标进行归一化，处理成0-1之间的数
        ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)

        # 得到每一层feature map对应的reference point，即ref，shape为[B, feat_W*feat_H, 2]
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)

    # 将所有尺度的feature map对应的reference point在第一维合并，得到[2, N, 2]
    reference_points = torch.cat(reference_points_list, 1)
    # 从[2, N, 2]扩充尺度到[2, N, num_level, 2] (x,y)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.contiguous()


@NECKS.register_module()
class MDA_TAM(BaseModule):
    #mutual deformable attention & temporal aggregation model
    def __init__(self,
                 single_frame_image_feat_channels,
                 adj_image_frames,
                 img_feat_channels,
                 pts_feat_channels,
                 out_channels,
                 spatial_size=[[180,180]],
                 conv_cfg=dict(type='Conv2d', bias=False),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super(MDA_TAM, self).__init__(init_cfg)
        self.embed_dim = out_channels
        self.spatial_size = spatial_size
        self.single_frame_image_feat_channels = single_frame_image_feat_channels
        self.adj_image_frames = adj_image_frames

        self.cur_img_proj_conv = build_conv_layer(conv_cfg, single_frame_image_feat_channels, out_channels, 1, stride=1, padding=0)
        self.pre_img_proj_conv = build_conv_layer(conv_cfg, img_feat_channels-single_frame_image_feat_channels, out_channels, 1, stride=1, padding=0, groups=adj_image_frames)#注意这里采用组卷积对历史帧分别处理
        self.pts_proj_conv = build_conv_layer(conv_cfg, pts_feat_channels, out_channels, 1, stride=1, padding=0)
        self.share_proj_conv = build_conv_layer(conv_cfg, out_channels + pts_feat_channels, out_channels, 1,stride=1, padding=0)

        self.PE = PositionEmbeddingSine(int(out_channels / 2))
        self.MSDA_img2img = MultiScaleDeformableAttention(embed_dims=out_channels, num_levels=1)
        self.MSDA_on_img = MultiScaleDeformableAttention(embed_dims=out_channels, num_levels=1)
        self.MSDA_on_pts = MultiScaleDeformableAttention(embed_dims=out_channels, num_levels=1)

    def forward(self, inputs):
        img_feats, pts_feats = inputs
        cur_img_feats = img_feats[:,:self.single_frame_image_feat_channels,:,:] #(B,80,H,W)
        pre_img_feats = img_feats[:,self.single_frame_image_feat_channels:,:,:] #(B,640,H,W)

        cur_img_proj = self.cur_img_proj_conv(cur_img_feats) #(B, C, H, W), C=256
        pre_img_proj = self.pre_img_proj_conv(pre_img_feats) #(B, C, H, W)

        B, _, _, _ = pts_feats.size()
        device = pts_feats.device
        pos_embed = self.PE(torch.ones(B, self.spatial_size[0][0], self.spatial_size[0][1]).bool())  # (B, e_d, H, W)
        pos_embed = pos_embed.permute(0, 2, 3, 1).contiguous().view(B, -1, self.embed_dim).permute(1, 0, 2).to(device)  # (hw,B,e_d)
        reference_points = get_reference_points(self.spatial_size, torch.ones(B, 1, 2)).to(device)  # (B, hw, 1, 2)

        curIMG_query = cur_img_proj.permute(0,2,3,1).contiguous().view(B,-1,self.embed_dim).permute(1,0,2) #(hw, B, e_d)
        preIMG_value = pre_img_proj.permute(0,2,3,1).contiguous().view(B,-1,self.embed_dim).permute(1,0,2) #(hw, B, e_d)
        img_feats_querylike = self.MSDA_img2img(query = curIMG_query,
                                         value = preIMG_value,
                                         query_pos = pos_embed,
                                         reference_points = reference_points,
                                         spatial_shapes = torch.Tensor(self.spatial_size).long().to(device),
                                         level_start_index = torch.Tensor([0]).long().to(device)
                                         )#(hw, B, e_d)
        img_feats_spatiallike = img_feats_querylike.permute(1,0,2).contiguous().view(B, self.spatial_size[0][0], self.spatial_size[0][1], self.embed_dim) #(B,H,W,C)
        img_feats_spatiallike = img_feats_spatiallike.permute(0,3,1,2).contiguous() #(B,C,H,W)

        share_feats = self.share_proj_conv(torch.cat([img_feats_spatiallike, pts_feats], axis=1))  # (B, C, H, W)
        proj_pts_feats = self.pts_proj_conv(pts_feats)  # (B, C, H, W)
        query = share_feats.permute(0, 2, 3, 1).contiguous().view(B, -1, self.embed_dim).permute(1, 0, 2)  # (hw, B, e_d)
        value_pts = proj_pts_feats.permute(0, 2, 3, 1).contiguous().view(B, -1, self.embed_dim).permute(1, 0, 2)  # (hw, B, e_d)

        share_ask_img = self.MSDA_on_img(query=query,
                                         value=img_feats_querylike,
                                         query_pos=pos_embed,
                                         reference_points=reference_points,
                                         spatial_shapes=torch.Tensor(self.spatial_size).long().to(device),
                                         level_start_index=torch.Tensor([0]).long().to(device)
                                         )
        share_ask_pts = self.MSDA_on_pts(query=query,
                                         value=value_pts,
                                         query_pos=pos_embed,
                                         reference_points=reference_points,
                                         spatial_shapes=torch.Tensor(self.spatial_size).long().to(device),
                                         level_start_index=torch.Tensor([0]).long().to(device)
                                         )  # (hw, B, e_d)
        out = (share_ask_img + share_ask_pts - query).permute(1,0,2).contiguous().view(B, self.spatial_size[0][0], self.spatial_size[0][1], self.embed_dim) #(B,H,W,C)
        return out.permute(0, 3, 1, 2).contiguous()

