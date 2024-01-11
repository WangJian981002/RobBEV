# Copyright (c) OpenMMLab. All rights reserved.
from .compose import Compose
from .dbsampler import DataBaseSampler
from .formating import Collect3D, DefaultFormatBundle, DefaultFormatBundle3D
from .loading import (LoadAnnotations3D, LoadAnnotationsBEVDepth,
                      LoadImageFromFileMono3D, LoadMultiViewImageFromFiles,
                      LoadPointsFromDict, LoadPointsFromFile,
                      LoadPointsFromMultiSweeps, NormalizePointsColor,
                      PointSegClassMapping, PointToMultiViewDepth,
                      PrepareImageInputs,PointTransform_with_BDA,
                      Points_fov_drop,Points_fov_drop_random,
                      Points_ring_index_select,Cam_drop,Random_remove_points_in_box,
                      MyResize,MyNormalize,MyPad,Cam_drop_transfusion,Cam_drop_seq,Cam_drop_test,Keep_Cam,
                      Single_Cam_drop_train,Single_Cam_drop_test,Cam_occlusion,Cam_occlusion_acc,PrepareImageInputsv2)
from .test_time_aug import MultiScaleFlipAug3D
# yapf: disable
from .transforms_3d import (AffineResize, BackgroundPointsFilter,
                            GlobalAlignment, GlobalRotScaleTrans,
                            IndoorPatchPointSample, IndoorPointSample,
                            MultiViewWrapper, ObjectNameFilter, ObjectNoise,
                            ObjectRangeFilter, ObjectSample, PointSample,
                            PointShuffle, PointsRangeFilter,
                            RandomDropPointsColor, RandomFlip3D,
                            RandomJitterPoints, RandomRotate, RandomShiftScale,
                            RangeLimitedRandomCrop, VoxelBasedPointSampler)

__all__ = [
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter', 'Collect3D',
    'Compose', 'LoadMultiViewImageFromFiles', 'LoadPointsFromFile',
    'DefaultFormatBundle', 'DefaultFormatBundle3D', 'DataBaseSampler',
    'NormalizePointsColor', 'LoadAnnotations3D', 'IndoorPointSample',
    'PointSample', 'PointSegClassMapping', 'MultiScaleFlipAug3D',
    'LoadPointsFromMultiSweeps', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'GlobalAlignment', 'IndoorPatchPointSample',
    'LoadImageFromFileMono3D', 'ObjectNameFilter', 'RandomDropPointsColor',
    'RandomJitterPoints', 'AffineResize', 'RandomShiftScale',
    'LoadPointsFromDict', 'MultiViewWrapper', 'RandomRotate',
    'RangeLimitedRandomCrop', 'PrepareImageInputs',
    'LoadAnnotationsBEVDepth', 'PointToMultiViewDepth','PointTransform_with_BDA',
    'Points_fov_drop', 'Points_fov_drop_random', 'Points_ring_index_select', 'Cam_drop',
    'Random_remove_points_in_box', 'MyResize', 'MyNormalize', 'MyPad', 'Cam_drop_transfusion',
    'Cam_drop_seq','Cam_drop_test','Keep_Cam','Single_Cam_drop_train','Single_Cam_drop_test',
    'Cam_occlusion','Cam_occlusion_acc', 'PrepareImageInputsv2'
]
