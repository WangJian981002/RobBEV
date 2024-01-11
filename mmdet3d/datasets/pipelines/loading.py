# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
import cv2
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES

@PIPELINES.register_module()
class MyResize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        imgs = results['img']
        results['img'] = [imgs[i] for i in range(len(imgs))]
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results[key][idx],
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][idx].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results[key][idx],
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                results[key][idx] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class MyNormalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                results[key][idx] = mmcv.imnormalize(results[key][idx], self.mean, self.std,
                                                     self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class MyPad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        #padded_img = None
        #print(results.get('img_fields', ['img']))
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                for idx in range(len(results[key])):
                    padded_img = mmcv.impad_to_multiple(
                        results[key][idx], self.size_divisor, pad_val=self.pad_val)
                    results[key][idx] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged'):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate([img, np.zeros_like(img[0:1280-886,:])], axis=0)
        return img

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        if self.img_scale is None:
            img = np.stack(
                [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        else:
            img = np.stack(
                [self.pad(mmcv.imread(name, self.color_type)) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = [1.0, 1.0]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)

@PIPELINES.register_module()
class Cam_drop_transfusion(object):
    #cam center angle ['CAM_FRONT_LEFT':145, 'CAM_FRONT':90, 'CAM_FRONT_RIGHT':35, 'CAM_BACK_LEFT':215, 'CAM_BACK':270, 'CAM_BACK_RIGHT':325]
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.lidar_drop_index2cam = {0:[1,3,5], 1:[3,4,5], 2:[2,3,4], 3:[0,1,5], 4:[0,1,2], 5:[0,2,4]}
        self.pad_value = 128
    def __call__(self, results):
        cam_drop_index = self.lidar_drop_index2cam[results['dropped_fov_index']]
        imgs= results['img']
        H,W,C = imgs[0].shape

        for i in cam_drop_index:
            if np.random.uniform() < self.drop_prob:
                imgs[i] = np.ones((H,W,C))*self.pad_value
        results['img'] = imgs
        return results


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        gt_bboxes_3d = results['lidar_ann_info']['gt_bboxes_3d']
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5))

        results['gt_bboxes_3d'] = gt_bboxes_3d
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['lidar_ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

def mmlabDeNormalize(img):
    from mmcv.image.photometric import imdenormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_bgr = True
    img = img.permute(1,2,0).numpy()
    img = imdenormalize(img, mean, std, to_bgr)
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][cam_name]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor

    def get_sensor2lidar_transformations(self,cam_info,cam_name):

        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran
        # global to ego lidar
        w, x, y, z = cam_info['ego2global_rotation']
        egolidar2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        egolidar2global_tran = torch.Tensor(cam_info['ego2global_translation'])
        egolidar2global = egolidar2global_rot.new_zeros((4, 4))
        egolidar2global[3, 3] = 1
        egolidar2global[:3, :3] = egolidar2global_rot
        egolidar2global[:3, -1] = egolidar2global_tran
        global2egolidar = egolidar2global.inverse()
        # ego lidar to lidar
        w, x, y, z = cam_info['lidar2ego_rotation']
        lidar2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        lidar2ego_tran = torch.Tensor(cam_info['lidar2ego_translation'])
        lidar2ego = lidar2ego_rot.new_zeros((4, 4))
        lidar2ego[3, 3] = 1
        lidar2ego[:3, :3] = lidar2ego_rot
        lidar2ego[:3, -1] = lidar2ego_tran
        ego2lidar = lidar2ego.inverse()

        sweepsensor2lidar = ego2lidar @ global2egolidar @ sweepego2global @ sweepsensor2sweepego

        return sweepsensor2lidar

    def get_sensor2lidar_transformations_cross_frame(self,cur_info,adj_info,cam_name):
        # sweep sensor to sweep ego
        w, x, y, z = adj_info['cams'][cam_name]['sensor2ego_rotation']
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            adj_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = adj_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            adj_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran
        # global to ego lidar
        w, x, y, z = cur_info['ego2global_rotation']
        egolidar2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        egolidar2global_tran = torch.Tensor(cur_info['ego2global_translation'])
        egolidar2global = egolidar2global_rot.new_zeros((4, 4))
        egolidar2global[3, 3] = 1
        egolidar2global[:3, :3] = egolidar2global_rot
        egolidar2global[:3, -1] = egolidar2global_tran
        global2egolidar = egolidar2global.inverse()
        # ego lidar to lidar
        w, x, y, z = cur_info['lidar2ego_rotation']
        lidar2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        lidar2ego_tran = torch.Tensor(cur_info['lidar2ego_translation'])
        lidar2ego = lidar2ego_rot.new_zeros((4, 4))
        lidar2ego[3, 3] = 1
        lidar2ego[:3, :3] = lidar2ego_rot
        lidar2ego[:3, -1] = lidar2ego_tran
        ego2lidar = lidar2ego.inverse()

        sweepsensor2lidar = ego2lidar @ global2egolidar @ sweepego2global @ sweepsensor2sweepego

        return sweepsensor2lidar

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sweepsensor2lidar = self.get_sensor2lidar_transformations(results['curr'], cam_name)
            sensor2sensor = torch.eye(4)
            rot = sweepsensor2lidar[:3, :3]
            tran = sweepsensor2lidar[:3, 3]


            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2lidar = self.get_sensor2lidar_transformations_cross_frame(results['curr'],adj_info,cam_name)
                    rot = adjsensor2lidar[:3, :3]
                    tran = adjsensor2lidar[:3, 3]
                    sensor2sensor = torch.eye(4)

                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)
        imgs = torch.stack(imgs)

        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results

@PIPELINES.register_module()
class PrepareImageInputsv2(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',
        opencv_pp=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.opencv_pp = opencv_pp

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        if not self.opencv_pp:
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        if self.opencv_pp:
            img = self.img_transform_core_opencv(img, post_rot, post_tran, crop)

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def img_transform_core_opencv(self, img, post_rot, post_tran, crop):
        img = np.array(img).astype(np.float32)
        img = cv2.warpAffine(img,
                             np.concatenate([post_rot,
                                             post_tran.reshape(2, 1)],
                                            axis=1),
                             (crop[2] - crop[0], crop[3] - crop[1]),
                             flags=cv2.INTER_LINEAR)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            random_crop_height = \
                self.data_config.get('random_crop_height', False)
            if random_crop_height:
                crop_h = int(np.random.uniform(max(0.3*newH, newH-fH),
                                               newH-fH))
            else:
                crop_h = \
                    int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
            if self.data_config.get('vflip', False) and np.random.choice([0, 1]):
                rotate += 180
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][cam_name]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor

    def get_sensor2lidar_transformations(self,cam_info,cam_name):

        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran
        # global to ego lidar
        w, x, y, z = cam_info['ego2global_rotation']
        egolidar2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        egolidar2global_tran = torch.Tensor(cam_info['ego2global_translation'])
        egolidar2global = egolidar2global_rot.new_zeros((4, 4))
        egolidar2global[3, 3] = 1
        egolidar2global[:3, :3] = egolidar2global_rot
        egolidar2global[:3, -1] = egolidar2global_tran
        global2egolidar = egolidar2global.inverse()
        # ego lidar to lidar
        w, x, y, z = cam_info['lidar2ego_rotation']
        lidar2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        lidar2ego_tran = torch.Tensor(cam_info['lidar2ego_translation'])
        lidar2ego = lidar2ego_rot.new_zeros((4, 4))
        lidar2ego[3, 3] = 1
        lidar2ego[:3, :3] = lidar2ego_rot
        lidar2ego[:3, -1] = lidar2ego_tran
        ego2lidar = lidar2ego.inverse()

        sweepsensor2lidar = ego2lidar @ global2egolidar @ sweepego2global @ sweepsensor2sweepego

        return sweepsensor2lidar

    def get_sensor2lidar_transformations_cross_frame(self,cur_info,adj_info,cam_name):
        # sweep sensor to sweep ego
        w, x, y, z = adj_info['cams'][cam_name]['sensor2ego_rotation']
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            adj_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = adj_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            adj_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran
        # global to ego lidar
        w, x, y, z = cur_info['ego2global_rotation']
        egolidar2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        egolidar2global_tran = torch.Tensor(cur_info['ego2global_translation'])
        egolidar2global = egolidar2global_rot.new_zeros((4, 4))
        egolidar2global[3, 3] = 1
        egolidar2global[:3, :3] = egolidar2global_rot
        egolidar2global[:3, -1] = egolidar2global_tran
        global2egolidar = egolidar2global.inverse()
        # ego lidar to lidar
        w, x, y, z = cur_info['lidar2ego_rotation']
        lidar2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        lidar2ego_tran = torch.Tensor(cur_info['lidar2ego_translation'])
        lidar2ego = lidar2ego_rot.new_zeros((4, 4))
        lidar2ego[3, 3] = 1
        lidar2ego[:3, :3] = lidar2ego_rot
        lidar2ego[:3, -1] = lidar2ego_tran
        ego2lidar = lidar2ego.inverse()

        sweepsensor2lidar = ego2lidar @ global2egolidar @ sweepego2global @ sweepsensor2sweepego

        return sweepsensor2lidar

    def photo_metric_distortion(self, img, pmd):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand() > pmd.get('rate', 1.0):
            return img

        img = np.array(img).astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-pmd['brightness_delta'],
                                      pmd['brightness_delta'])
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                          pmd['contrast_upper'])
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(pmd['saturation_lower'],
                                             pmd['saturation_upper'])

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-pmd['hue_delta'], pmd['hue_delta'])
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                          pmd['contrast_upper'])
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return Image.fromarray(img.astype(np.uint8))

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sweepsensor2lidar = self.get_sensor2lidar_transformations(results['curr'], cam_name)
            sensor2sensor = torch.eye(4)
            rot = sweepsensor2lidar[:3, :3]
            tran = sweepsensor2lidar[:3, 3]


            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            if self.is_train and self.data_config.get('pmd', None) is not None:
                img = self.photo_metric_distortion(img, self.data_config['pmd'])

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    if self.opencv_pp:
                        img_adjacent = \
                            self.img_transform_core_opencv(
                                img_adjacent,
                                post_rot[:2, :2],
                                post_tran[:2],
                                crop)
                    else:
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2lidar = self.get_sensor2lidar_transformations_cross_frame(results['curr'],adj_info,cam_name)
                    rot = adjsensor2lidar[:3, :3]
                    tran = adjsensor2lidar[:3, 3]
                    sensor2sensor = torch.eye(4)

                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)
        imgs = torch.stack(imgs)

        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results

@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        if 'lidar_ann_info' not in results.keys():
            imgs, rots, trans, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]

            bda_rot = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot)
            return results

        lidar_ann_info = results['lidar_ann_info']
        gt_boxes, gt_labels = torch.from_numpy(lidar_ann_info['gt_bboxes_3d']).to(torch.float32), torch.from_numpy(lidar_ann_info['gt_labels_3d'])
        #gt_boxes, gt_labels = results['ann_infos']
        #gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)

        if 'points' in results:
            points = results['points']
            points.tensor[:,:3] = (bda_rot @ points.tensor[:,:3].unsqueeze(-1)).squeeze(-1)
            results['points'] = points

        return results

@PIPELINES.register_module()
class PointTransform_with_BDA(object):

    def __init__(self):
        pass
    def __call__(self, results):
        bda_rot = results['img_inputs'][-1]
        points = results['points']
        points.tensor[:, :3] = (bda_rot @ points.tensor[:, :3].unsqueeze(-1)).squeeze(-1)
        results['points'] = points
        return results

@PIPELINES.register_module()
class Points_fov_drop(object):
    #对drop_range中每个fov，依照drop_prob随机进行drop，相互独立
    def __init__(self, drop_range, drop_prob):
        self.drop_range = drop_range
        self.drop_prob = drop_prob
        assert len(drop_range) == len(drop_prob)
    def __call__(self, results):
        points = results['points']
        x = points.tensor[:, 0]
        maskx = (x>=0)
        p1s = points.tensor[maskx]
        x1 = p1s[:, 0]
        y1 = p1s[:, 1]
        x1 = torch.clamp(x1,min=1e-9)
        angle1 = torch.arctan(y1/x1) #(-pi/2, pi/2)
        angle1 = (angle1 + 2*torch.pi) % (2*torch.pi) #[0,pi/2) and (1.5pi,2pi)

        p2s = points.tensor[~maskx]
        x2 = p2s[:, 0]
        y2 = p2s[:, 1]
        x2 = torch.clamp(x2, max=-1e-9)
        angle2 = torch.arctan(y2/x2) + torch.pi #(0.5pi,1.5pi)

        ps = torch.cat([p1s,p2s],dim=0)
        angle = torch.cat([angle1,angle2]) #[0,2pi)

        keep_mask = torch.ones(len(ps))
        for i in range(len(self.drop_range)):
            a_range = self.drop_range[i]
            a_prob = self.drop_prob[i]
            if np.random.uniform() < a_prob:
                mask = (angle > a_range[0]) & (angle < a_range[1])
                keep_mask = keep_mask*(~mask).float()
        points.tensor = ps[keep_mask.bool()]
        results['points'] = points
        return results

@PIPELINES.register_module()
class Points_fov_drop_random(object):
    #对drop_range中每个fov,依照drop_prob随机drop一个
    #cam center angle ['CAM_FRONT_LEFT':145, 'CAM_FRONT':90, 'CAM_FRONT_RIGHT':35, 'CAM_BACK_LEFT':215, 'CAM_BACK':270, 'CAM_BACK_RIGHT':325]
    def __init__(self, drop_range=[[(85,205)], [(30,150)], [(335,360),(0,95)], [(155,275)], [(210,330)], [(265,360),(0,25)]], drop_prob=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
        self.drop_range = drop_range
        self.drop_prob = drop_prob
        assert len(drop_range) == len(drop_prob)
    def __call__(self, results):
        points = results['points']
        x = points.tensor[:, 0]
        maskx = (x>=0)
        p1s = points.tensor[maskx]
        x1 = p1s[:, 0]
        y1 = p1s[:, 1]
        x1 = torch.clamp(x1,min=1e-9)
        angle1 = torch.arctan(y1/x1) #(-pi/2, pi/2)
        angle1 = (angle1 + 2*torch.pi) % (2*torch.pi) #[0,pi/2) and (1.5pi,2pi)

        p2s = points.tensor[~maskx]
        x2 = p2s[:, 0]
        y2 = p2s[:, 1]
        x2 = torch.clamp(x2, max=-1e-9)
        angle2 = torch.arctan(y2/x2) + torch.pi #(0.5pi,1.5pi)

        ps = torch.cat([p1s,p2s],dim=0)
        angle = torch.cat([angle1,angle2]) #[0,2pi)

        drop_index = np.random.choice([0,1,2,3,4,5], 1, p=self.drop_prob)[0]
        d_range = self.drop_range[drop_index]
        keep_mask = torch.ones(len(ps))
        for d_range_i in d_range:
            min_angle = (d_range_i[0] / 180.)*torch.pi
            max_angle = (d_range_i[1] / 180.)*torch.pi
            mask = (angle > min_angle) & (angle < max_angle)
            keep_mask = keep_mask * (~mask).float()
        points.tensor = ps[keep_mask.bool()]
        results['points'] = points
        results['dropped_fov_index'] = drop_index
        return results

@PIPELINES.register_module()
class Cam_drop(object):
    #cam center angle ['CAM_FRONT_LEFT':145, 'CAM_FRONT':90, 'CAM_FRONT_RIGHT':35, 'CAM_BACK_LEFT':215, 'CAM_BACK':270, 'CAM_BACK_RIGHT':325]
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.lidar_drop_index2cam = {0:[2,4,5], 1:[3,4,5], 2:[0,3,4], 3:[1,2,5], 4:[0,1,2], 5:[0,1,3]}
        self.pad_value = torch.Tensor([0.074,0.205,0.426]).view(3,1,1)
    def __call__(self, results):
        cam_drop_index = self.lidar_drop_index2cam[results['dropped_fov_index']]
        imgs, *k = results['img_inputs']

        for i in cam_drop_index:
            if np.random.uniform() < self.drop_prob:
                imgs[i,:,:,:] = self.pad_value
        results['img_inputs'] = (imgs, *k)
        return results

@PIPELINES.register_module()
class Points_ring_index_select(object):

    def __init__(self, reduce_ratio=[1, 2, 4], reduce_prob = [0.5, 0.375, 0.125]):
        self.reduce_prob = reduce_prob
        self.reduce_ratio = reduce_ratio
    def __call__(self, results):
        red = np.random.choice(self.reduce_ratio, 1, p=self.reduce_prob)[0]
        if red == 1:
            return results
        else:
            points = results['points']
            r_index = points.tensor[:, 3]
            mask = (r_index % red) == np.random.choice(red)
            points.tensor = points.tensor[mask]
            results['points'] = points
            return results

@PIPELINES.register_module()
class Random_remove_points_in_box(object):
    def __init__(self, drop_prob):
        self.drop_prob = drop_prob

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points, boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, results):
        gt_box = results['gt_bboxes_3d'].tensor.numpy() #(N,7)
        N = gt_box.shape[0]
        remove_index = np.random.choice(N, int(N*self.drop_prob), replace=False)
        remove_box = gt_box[remove_index]

        if len(remove_box) > 0:
            points = results['points']
            removed_points_coords = self.remove_points_in_boxes(points.tensor.numpy(), remove_box)
            points.tensor = torch.from_numpy(removed_points_coords)
            results['points'] = points
        return results

@PIPELINES.register_module()
class Cam_drop_seq(object):
    #cam name ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    def __init__(self, num_adj_frame=8, drop_prob=0.5, seq_drop_prob=0.5):
        self.num_adj_frame = num_adj_frame
        self.drop_prob = drop_prob
        self.seq_drop_prob = seq_drop_prob
        self.pad_value = torch.Tensor([0.074,0.205,0.426]).view(3,1,1)
    def __call__(self, results):
        imgs, *k = results['img_inputs']
        for i in range(len(['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])):
            if np.random.uniform() < self.drop_prob:
                if np.random.uniform() < self.seq_drop_prob:
                    imgs[i*(self.num_adj_frame+1):(i+1)*(self.num_adj_frame+1),:,:,:] = self.pad_value.unsqueeze(0).repeat((self.num_adj_frame+1), 1,1,1)
                else:
                    imgs[i*(self.num_adj_frame+1),:,:,:] = self.pad_value
        results['img_inputs'] = (imgs, *k)
        return results

@PIPELINES.register_module()
class Cam_drop_test(object):
    #cam name ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    def __init__(self, num_adj_frame=8,drop_cam_list=[],is_seq=False):
        self.num_adj_frame = num_adj_frame
        self.drop_cam_list = drop_cam_list
        self.is_seq = is_seq
        self.cam_name_idx = {'CAM_FRONT_LEFT':0, 'CAM_FRONT':1, 'CAM_FRONT_RIGHT':2, 'CAM_BACK_LEFT':3, 'CAM_BACK':4, 'CAM_BACK_RIGHT':5}
        self.pad_value = torch.Tensor([0.074,0.205,0.426]).view(3,1,1)
    def __call__(self, results):
        imgs, *k = results['img_inputs']
        for drop_cam_name in self.drop_cam_list:
            drop_i = self.cam_name_idx[drop_cam_name]
            if self.is_seq:
                imgs[drop_i * (self.num_adj_frame + 1):(drop_i + 1) * (self.num_adj_frame + 1), :, :, :] = self.pad_value.unsqueeze(0).repeat((self.num_adj_frame + 1), 1, 1, 1)
            else:
                imgs[drop_i*(self.num_adj_frame+1),:,:,:] = self.pad_value
        results['img_inputs'] = (imgs, *k)
        return results


@PIPELINES.register_module()
class Keep_Cam(object):
    #cam name ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    def __init__(self, num_adj_frame=8,keep_cam_list=[],is_seq=False):
        self.num_adj_frame = num_adj_frame
        self.keep_cam_list = keep_cam_list
        self.is_seq = is_seq
        self.cam_name_idx = {'CAM_FRONT_LEFT':0, 'CAM_FRONT':1, 'CAM_FRONT_RIGHT':2, 'CAM_BACK_LEFT':3, 'CAM_BACK':4, 'CAM_BACK_RIGHT':5}
        self.pad_value = torch.Tensor([0.074,0.205,0.426]).view(3,1,1)
    def __call__(self, results):
        imgs, *k = results['img_inputs']
        N, _, H, W = imgs.size()
        new_imgs = self.pad_value.unsqueeze(0).repeat(N, 1, H, W)
        for keep_cam_name in self.keep_cam_list:
            keep_i = self.cam_name_idx[keep_cam_name]
            if self.is_seq:
                new_imgs[keep_i * (self.num_adj_frame + 1):(keep_i + 1) * (self.num_adj_frame + 1),:,:,:] = imgs[keep_i * (self.num_adj_frame + 1):(keep_i + 1) * (self.num_adj_frame + 1),:,:,:]
            else:
                new_imgs[keep_i*(self.num_adj_frame+1),:,:,:] = imgs[keep_i*(self.num_adj_frame+1),:,:,:]
        results['img_inputs'] = (new_imgs, *k)
        return results

@PIPELINES.register_module()
class Single_Cam_drop_train(object):
    #cam name ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    def __init__(self, num_drop):
        self.num_drop = num_drop
        self.pad_value = torch.Tensor([0.074,0.205,0.426]).view(3,1,1)
    def __call__(self, results):
        imgs, *k = results['img_inputs']
        drop_index = np.random.choice(6,self.num_drop,replace=False)
        for i in drop_index:
            imgs[i,:,:,:] = self.pad_value
        results['img_inputs'] = (imgs, *k)
        return results

@PIPELINES.register_module()
class Single_Cam_drop_test(object):
    #cam name ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    def __init__(self, drop_cam_list):
        self.drop_cam_list = drop_cam_list
        self.cam_name_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3,
                             'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
        self.pad_value = torch.Tensor([0.074,0.205,0.426]).view(3,1,1)
    def __call__(self, results):
        imgs, *k = results['img_inputs']
        for cam_name in self.drop_cam_list:
            imgs[self.cam_name_idx[cam_name], :, :, :] = self.pad_value
        results['img_inputs'] = (imgs, *k)
        return results


@PIPELINES.register_module()
class Cam_occlusion(object):
    #cam name ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    def __init__(self, num_adj_frame=8,is_seq=False,mud_dir='demo/Mud_Mask_selected/'):
        self.num_adj_frame = num_adj_frame
        self.is_seq = is_seq
        self.mud_dir = mud_dir

    def put_mask_on_img(self, img, mask):
        #img (H,W,3) bgr
        h, w = img.shape[:2]
        mask = np.rot90(mask)
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = mask / 255
        alpha = np.power(alpha, 3)
        img_with_mask = alpha * img + (1 - alpha) * mask
        return img_with_mask

    def __call__(self, results):
        imgs, *k = results['img_inputs']
        for cam_i in range(6):
            mask_idx = np.random.choice(list(range(1,17)), 1)[0]
            mask_dir = self.mud_dir + "mask_" + str(mask_idx) + '.jpg'
            mask = mmcv.imread(mask_dir, 'unchanged')
            if self.is_seq:
                for j in range(cam_i*(self.num_adj_frame+1), (cam_i+1)*(self.num_adj_frame+1)):
                    img = imgs[j,:,:,:]
                    img = mmlabDeNormalize(img) #rgb
                    img = img[:,:,::-1]
                    img = self.put_mask_on_img(img, mask)
                    img = img[:,:,::-1]
                    imgs[j, :, :, :] = mmlabNormalize(img)
            else:
                img = imgs[cam_i,:,:,:]
                img = mmlabDeNormalize(img)  # rgb
                img = img[:,:,::-1]
                img = self.put_mask_on_img(img, mask)
                img = img[:,:,::-1]
                imgs[cam_i, :, :, :] = mmlabNormalize(img)

        results['img_inputs'] = (imgs, *k)
        return results


@PIPELINES.register_module()
class Cam_occlusion_acc(object):
    #cam name ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    def __init__(self, num_adj_frame=8,is_seq=False,mud_dir='demo/Mud_Mask_selected/'):
        self.num_adj_frame = num_adj_frame
        self.is_seq = is_seq
        self.mud_dir = mud_dir
        self.mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)#bgr下
        self.std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)


    def __call__(self, results):
        imgs, *k = results['img_inputs']
        H,W = imgs.size()[2:]

        mask_list = []
        alpha_list = []
        for cam_i in range(6):
            mask_idx = np.random.choice(list(range(1,17)), 1)[0]
            mask_dir = self.mud_dir + "mask_" + str(mask_idx) + '.jpg'
            mask = mmcv.imread(mask_dir, 'unchanged')
            mask = np.rot90(mask)
            mask = mmcv.imresize(mask, (W, H), return_scale=False)
            alpha = mask / 255
            alpha = np.power(alpha, 3)

            if self.is_seq:
                mask = np.tile(mask[np.newaxis,...],(1+self.num_adj_frame,1,1,1)) #(9,H,W,3)
                alpha = np.tile(alpha[np.newaxis,...],(1+self.num_adj_frame,1,1,1)) #(9,H,W,3)
                mask_list.append(mask)
                alpha_list.append(alpha)
            else:
                mask = mask[np.newaxis,...] #(1,H,W,3)
                alpha = alpha[np.newaxis,...] #(1,H,W,3)
                mask_list.append(mask)
                alpha_list.append(alpha)
        mask_list = np.concatenate(mask_list,axis=0) #(N,H,W,3)
        alpha_list = np.concatenate(alpha_list, axis=0)#(N,H,W,3)

        imgs = imgs.permute(0,2,3,1).numpy()#(N,H,W,3) BGR
        imgs = (imgs*self.std) + self.mean
        imgs = alpha_list*imgs + (1 - alpha_list)*mask_list
        imgs = (imgs-self.mean)/self.std
        imgs = torch.from_numpy(imgs.astype(np.float32)).permute(0,3,1,2) #(N,3,H,W) BGR

        results['img_inputs'] = (imgs, *k)
        return results