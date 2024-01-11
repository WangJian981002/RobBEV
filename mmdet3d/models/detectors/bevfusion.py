# Copyright (c) Phigent Robotics. All rights reserved.
import mmcv
import torch
from os import path as osp
from torch import nn as nn
import torch.nn.functional as F
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class BEVFusion(CenterPoint):

    def __init__(self, img_view_transformer, fusion_neck, **kwargs):
        super(BEVFusion, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.fusion_layer = builder.build_neck(fusion_neck)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])

        return x, depth

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        return x

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points)
        fused_feats = self.fusion_layer([img_feats,pts_feats])

        x = self.pts_backbone(fused_feats)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        fused_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)


        losses = dict()
        losses_pts = self.forward_pts_train(fused_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""


        fused_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for i in range(len(img_metas))]
        if fused_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                fused_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    '''
    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs
    '''

@DETECTORS.register_module()
class BEVFusion4D(BEVFusion):
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVFusion4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev

    @force_fp32()
    def shift_feature(self, input, trans, rots, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _ = trans[0].shape

        # generate grid
        xs = torch.linspace(
            0, w - 1, w, dtype=input.dtype,
            device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(
            0, h - 1, h, dtype=input.dtype,
            device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c02l0[:, :, :3, :3] = rots[0][:, 0:1, :, :]
        c02l0[:, :, :3, 3] = trans[0][:, 0:1, :]
        c02l0[:, :, 3, 3] = 1

        # transformation from adjacent camera frame to current ego frame
        c12l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c12l0[:, :, :3, :3] = rots[1][:, 0:1, :, :]
        c12l0[:, :, :3, 3] = trans[1][:, 0:1, :]
        c12l0[:, :, 3, 3] = 1

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...],
                       intrins, post_rots, post_trans, bda[0:1,
                                                           ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [trans_curr, trans_prev],
                               [rots_curr, rots_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, rots, trans, intrins, post_rots, post_trans, bda = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    rot, tran = rots[0], trans[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert rots[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            trans_curr = trans[0].repeat(self.num_frame - 1, 1, 1)
            rots_curr = rots[0].repeat(self.num_frame - 1, 1, 1, 1)
            trans_prev = torch.cat(trans[1:], dim=0)
            rots_prev = torch.cat(rots[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [
                imgs[0], rots_curr, trans_curr, intrins[0], rots_prev,
                trans_prev, post_rots[0], post_trans[0], bda_curr
            ]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [trans[0], trans[adj_id]],
                                       [rots[0], rots[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        #x = self.bev_encoder(bev_feat)
        return bev_feat, depth_list[0]

@DETECTORS.register_module()
class BEVFusion4D_Depth(BEVFusion4D):

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points)
        fused_feats = self.fusion_layer([img_feats, pts_feats])

        x = self.pts_backbone(fused_feats)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x, depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):

        fused_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(fused_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""


        fused_feats, depth = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for i in range(len(img_metas))]
        if fused_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                fused_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        return bbox_list

@DETECTORS.register_module()
class BEVFusion_tf(MVXTwoStageDetector):

    def __init__(self, img_view_transformer, fusion_neck, **kwargs):
        super(BEVFusion_tf, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.fusion_layer = builder.build_neck(fusion_neck)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])

        return x, depth

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        return x

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points)
        fused_feats = self.fusion_layer([img_feats,pts_feats])

        x = self.pts_backbone(fused_feats)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        fused_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)


        losses = dict()
        losses_pts = self.forward_pts_train(fused_feats, None, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses


    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""


        fused_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for i in range(len(img_metas))]
        if fused_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                fused_feats, None, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

@DETECTORS.register_module()
class BEVFusion4D_tf(BEVFusion_tf):
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVFusion4D_tf, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev

    @force_fp32()
    def shift_feature(self, input, trans, rots, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _ = trans[0].shape

        # generate grid
        xs = torch.linspace(
            0, w - 1, w, dtype=input.dtype,
            device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(
            0, h - 1, h, dtype=input.dtype,
            device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c02l0[:, :, :3, :3] = rots[0][:, 0:1, :, :]
        c02l0[:, :, :3, 3] = trans[0][:, 0:1, :]
        c02l0[:, :, 3, 3] = 1

        # transformation from adjacent camera frame to current ego frame
        c12l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c12l0[:, :, :3, :3] = rots[1][:, 0:1, :, :]
        c12l0[:, :, :3, 3] = trans[1][:, 0:1, :]
        c12l0[:, :, 3, 3] = 1

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                [True, True, False, True], :][:, :, :, :,
                [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...],
                       intrins, post_rots, post_trans, bda[0:1,
                                                       ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [trans_curr, trans_prev],
                               [rots_curr, rots_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, rots, trans, intrins, post_rots, post_trans, bda = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    rot, tran = rots[0], trans[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert rots[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            trans_curr = trans[0].repeat(self.num_frame - 1, 1, 1)
            rots_curr = rots[0].repeat(self.num_frame - 1, 1, 1, 1)
            trans_prev = torch.cat(trans[1:], dim=0)
            rots_prev = torch.cat(rots[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [
                imgs[0], rots_curr, trans_curr, intrins[0], rots_prev,
                trans_prev, post_rots[0], post_trans[0], bda_curr
            ]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [trans[0], trans[adj_id]],
                                       [rots[0], rots[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        #x = self.bev_encoder(bev_feat)
        return bev_feat, depth_list[0]


