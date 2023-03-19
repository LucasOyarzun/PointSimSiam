from .simsiam import SimSiam
from .backbones import PointNetfeat
import torch


def get_backbone(backbone, cut_cl_head=True):
    backbone = eval(f"{backbone}()")
    if cut_cl_head:
        pass
    return backbone


def get_model(config):
    if config.name == "simsiam":
        model = SimSiam(get_backbone(config.backbone))
        if config.proj_layers is not None:
            model.projector.set_layers(config.proj_layers)
    else:
        raise NotImplementedError
    model = torch.nn.DataParallel(model)
    return model
