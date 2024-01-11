# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp
from collections import OrderedDict

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    #load pretrained checkpoints
    if 'load_img_from' in cfg:
        print(cfg.load_img_from)
        checkpoint= torch.load(cfg.load_img_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        ckpt = state_dict

        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('backbone'):
                new_v = v
                new_k = k.replace('backbone.', 'img_backbone.')
            else:
                continue
            new_ckpt[new_k] = new_v
        model.load_state_dict(new_ckpt, strict=False)
        logger.info(f"load img pretrained weight from: "+cfg.load_img_from)
        for k in new_ckpt.keys():
            logger.info(k)



    if 'img_pre_weight' in cfg:
        #print(cfg.img_pre_weight)
        checkpoint = torch.load(cfg.img_pre_weight, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('img_backbone'):
                new_v = v
                new_k = k
            elif k.startswith('img_neck'):
                new_v = v
                new_k = k
            elif k.startswith('img_view_transformer'):
                new_v = v
                new_k = k
            else:
                continue
            new_ckpt[new_k] = new_v
        logger.info(f"load img stream weight from: "+cfg.img_pre_weight)
        for k in new_ckpt.keys():
            logger.info(k)
        model.load_state_dict(new_ckpt, strict=False)

    if 'pts_pre_weight' in cfg:
        checkpoint = torch.load(cfg.pts_pre_weight, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict

        #for k in list(ckpt.keys()):
        #    if k.startswith('pts_bbox_head'):
        #        del ckpt[k]
        logger.info(f"load pts stream weight from: " + cfg.pts_pre_weight)
        for k in ckpt.keys():
            logger.info(k)
        model.load_state_dict(ckpt, strict=False)
        '''
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('pts_middle_encoder'):
                new_v = v
                new_k = k
            elif k.startswith('pts_backbone'):
                new_v = v
                new_k = k
            elif k.startswith('pts_neck'):
                new_v = v
                new_k = k
            elif k.startswith('pts_bbox_head'):
                new_v = v
                new_k = k
            else:
                continue
            new_ckpt[new_k] = new_v
        
        logger.info(f"load pts stream weight from: " + cfg.pts_pre_weight)
        for k in new_ckpt.keys():
            logger.info(k)
        
        model.load_state_dict(ckpt, strict=False) ###这里有一点问题，读new_ckpt会报维数错误
        '''

    if 'load_complete_weight' in cfg:
        checkpoint = torch.load(cfg.load_complete_weight, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        logger.info(f"********load model weight from: " + cfg.load_complete_weight)

    if 'freeze_image_backbone' in cfg and cfg['freeze_image_backbone'] is True:
        for name, param in model.named_parameters():
            if 'img_backbone' in name:
                param.requires_grad = False
            if 'img_neck' in name:
                param.requires_grad = False
            if 'img_view_transformer' in name:
                param.requires_grad = False

        from torch import nn

        def fix_bn(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        model.img_backbone.apply(fix_bn)
        model.img_neck.apply(fix_bn)
        model.img_view_transformer.apply(fix_bn)
        logger.info('freeze_image_backbone')

    if 'freeze_lidar_backbone' in cfg and cfg['freeze_lidar_backbone'] is True:
        for name, param in model.named_parameters():
            if 'pts_middle_encoder' in name:
                param.requires_grad = False
            if 'pts_backbone' in name:
                param.requires_grad = False
            if 'pts_neck' in name:
                param.requires_grad = False
            if 'pts_bbox_head' in name:
                param.requires_grad = False

        from torch import nn

        def fix_bn(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        model.pts_voxel_layer.apply(fix_bn)
        model.pts_voxel_encoder.apply(fix_bn)
        model.pts_middle_encoder.apply(fix_bn)
        model.pts_backbone.apply(fix_bn)
        model.pts_neck.apply(fix_bn)
        logger.info('freeze_lidar_backbone')

    # only for transfusion
    if 'freeze_lidar_components' in cfg and cfg['freeze_lidar_components'] is True:

        for name, param in model.named_parameters():
            if 'pts' in name and 'pts_bbox_head' not in name:
                param.requires_grad = False
            if 'pts_bbox_head.decoder.0' in name:
                param.requires_grad = False
            if 'pts_bbox_head.shared_conv' in name and 'pts_bbox_head.shared_conv_img' not in name:
                param.requires_grad = False
            if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
                param.requires_grad = False
            if 'pts_bbox_head.prediction_heads.0' in name:
                param.requires_grad = False
            if 'pts_bbox_head.class_encoding' in name:
                param.requires_grad = False

        from torch import nn

        def fix_bn(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        model.pts_voxel_layer.apply(fix_bn)
        model.pts_voxel_encoder.apply(fix_bn)
        model.pts_middle_encoder.apply(fix_bn)
        model.pts_backbone.apply(fix_bn)
        model.pts_neck.apply(fix_bn)
        model.pts_bbox_head.heatmap_head.apply(fix_bn)
        model.pts_bbox_head.shared_conv.apply(fix_bn)
        model.pts_bbox_head.class_encoding.apply(fix_bn)
        model.pts_bbox_head.decoder[0].apply(fix_bn)
        model.pts_bbox_head.prediction_heads[0].apply(fix_bn)

    if 'load_and_freeze_img_components' in cfg:
        checkpoint = torch.load(cfg.load_and_freeze_img_components, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        ckpt = state_dict

        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('backbone'):
                new_v = v
                new_k = 'img_'+ k
            elif k.startswith('neck'):
                new_v = v
                new_k = 'img_' + k
            else:
                continue
            new_ckpt[new_k] = new_v
        logger.info(f"load img stream weight from: " + cfg.load_and_freeze_img_components)
        for k in new_ckpt.keys():
            logger.info(k)
        model.load_state_dict(new_ckpt, strict=False)

        for name, param in model.named_parameters():
            if 'img_backbone' in name:
                param.requires_grad = False
            if 'img_neck' in name:
                param.requires_grad = False

        from torch import nn

        def fix_bn(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        model.img_backbone.apply(fix_bn)
        model.img_neck.apply(fix_bn)

    if 'load_tf_complete_weight' in cfg:
        checkpoint = torch.load(cfg.load_tf_complete_weight, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        logger.info(f"********load model weight from: " + cfg.load_tf_complete_weight)

    logger.info(f"param need to update:")
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            logger.info(name)

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
