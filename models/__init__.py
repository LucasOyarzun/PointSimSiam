from .simsiam import SimSiam
from .backbones import PointNetfeat


def get_backbone(backbone, cut_cl_head=True):
    backbone = eval(f"{backbone}()")

    if cut_cl_head:
        pass

    return backbone


def get_model(model_args):
    if model_args.name == "simsiam":
        model = SimSiam(get_backbone(model_args.backbone))
        if model_args.proj_layers is not None:
            model.projector.set_layers(model_args.proj_layers)
    else:
        raise NotImplementedError

    return model
