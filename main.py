import torch
from utils.parser import parse_args
from tools import Simsiam_pretrain_run_net


def main():
    # Parse arguments (console + config file)
    config = parse_args()

    # CUDA
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    Simsiam_pretrain_run_net(config)


if __name__ == "__main__":
    main()
