import os
import torch
from utils import parser, dist_utils, misc
from tools import Simsiam_pretrain_run_net
from tensorboardX import SummaryWriter


def main():
    # Parse arguments (console + config file)
    config = parser.parse_args()
    # CUDA
    config.use_gpu = torch.cuda.is_available()
    if config.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if config.launcher == "none":
        config.distributed = False
    else:
        config.distributed = True
        dist_utils.init_dist(config.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        config.world_size = world_size

    # define the tensorboard writer
    if not config.test:
        if config.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(config.tfboard_path, "train"))
            val_writer = SummaryWriter(os.path.join(config.tfboard_path, "test"))
        else:
            train_writer = None
            val_writer = None

    # batch size
    if config.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if hasattr(config, "extra_train"):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if hasattr(config, "test"):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if hasattr(config, "extra_train"):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if hasattr(config, "test"):
            config.dataset.test.others.bs = config.total_bs
    # Train
    Simsiam_pretrain_run_net(config)


if __name__ == "__main__":
    main()
