import argparse
import yaml
import re

### Arguments and configuration parser


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    # Add arguments here
    parser.add_argument("--config", type=str, help="Configuration file, e.g. config/pretrain.yaml")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--finetune', type=bool, default=False, help="If its a finetune run, default=False: pretrain")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        for key, value in Namespace(
            yaml.load(f, Loader=yaml.FullLoader)
        ).__dict__.items():
            vars(args)[key] = value

    return args
