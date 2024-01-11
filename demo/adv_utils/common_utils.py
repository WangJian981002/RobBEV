import logging
import os
import pickle
import random
import shutil
import subprocess
#import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_numpy_to_torch(x):
    # 检测输入数据是否是numpy格式，如果是，则转换为torch格式
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    # 首先，numpy格式数据转换为torch格式
    val, is_numpy = check_numpy_to_torch(val)
    # 将方位角限制在[-pi/2, pi/2] 返回 [-1.57, 1.57]弧度之间的方位角
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    # 首先利用torch.from_numpy().float将numpy转化为torch
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    # 构造旋转矩阵batch个
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    # 对点云坐标进行旋转
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    # 将旋转后的点云与原始点云拼接
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    # 将点云转化为numpy格式，并返回
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    # 根据点云的范围产生mask，过滤点云
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    # 获取每个voxel的xyz坐标
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()
    # 得到经过指定下采样后，每个voxel的长宽高
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    # [0, -40, -3]
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    # 得到每个voxel中心在点云中的xyz坐标 shape（N， 3）
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    # https://www.cnblogs.com/yyds/p/6901864.html
    # 初始化日志记录器：logger = logging.getLogger(logger_name)
    # 当直接执行一段脚本的时候，这段脚本的 __name__变量等于 '__main__'
    # 当这段脚本被导入其他程序的时候，__name__ 变量等于脚本本身的名字
    # 日志器（logger）是入口，真正干活儿的是处理器（handler），处理器（handler）还可以
    # 通过过滤器（filter）和格式器（formatter）对要输出的日志内容做过滤和格式化等处理操作
    logger = logging.getLogger(__name__)
    # 设置日志级别为INFO，即只有日志级别大于等于INFO的日志才会输出
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    # handler：将日志记录（log record）发送到合适的目的地（destination），比如文件，socket和控制台等。
    # 一个logger对象可以通过addHandler方法添加0到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示。
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        # 将日志消息发送到文件
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    # 通过类别获取被保留的索引
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds  # 将非空的voxel找出来，在ret中设置为该voxel的索引，空的voxel数值设置为-1
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device  # 获取该tensor所在设备
    batch_size = sparse_tensor.batch_size  # batch size
    spatial_shape = sparse_tensor.spatial_shape  # 稀疏卷积下采样后voxel特征层的尺度大小
    indices = sparse_tensor.indices.long()  # batch_idx, z, y, x
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)  # shape (num_of_voxel,)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape) # shape =  output_shape
    return v2pinds_tensor


def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
