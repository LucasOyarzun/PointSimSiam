import torch
from utils import parser
from tools import Simsiam_pretrain_run_net, Simsiam_finetune_run_net


def main():
    # Parse arguments (console + config file)
    config = parser.parse_args()

    # CUDA
    config.use_gpu = torch.cuda.is_available()

    # Finetune
    if config.finetune:
        Simsiam_finetune_run_net(config)
    else:
        # Pretrain
        Simsiam_pretrain_run_net(config)

    


if __name__ == "__main__":
    main()