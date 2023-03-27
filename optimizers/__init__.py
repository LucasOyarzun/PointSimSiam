import torch
from timm.scheduler import CosineLRScheduler
from utils.misc import *


def get_optimizer_sche(model, config):
    optimizer_config = config.optimizer

    predictor_prefix = ("module.predictor", "predictor")
    parameters = [
        {
            "name": "base",
            "params": [
                param
                for name, param in model.named_parameters()
                if not name.startswith(predictor_prefix)
            ],
            "lr": optimizer_config.kwargs.lr,
        },
        {
            "name": "predictor",
            "params": [
                param
                for name, param in model.named_parameters()
                if name.startswith(predictor_prefix)
            ],
            "lr": optimizer_config.kwargs.lr,
        },
    ]

    if optimizer_config.name == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            **optimizer_config.kwargs.__dict__,
            nesterov=True,
        )
    elif optimizer_config.name == "Adam":
        optimizer = torch.optim.Adam(parameters, **optimizer_config.kwargs.__dict__)
    elif optimizer_config.name == "AdamW":

        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if (
                    len(param.shape) == 1
                    or name.endswith(".bias")
                    or "token" in name
                    or name in skip_list
                ):
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {"params": no_decay, "weight_decay": 0.0},
                {"params": decay, "weight_decay": weight_decay},
            ]

        param_groups = add_weight_decay(
            model, weight_decay=optimizer_config.kwargs.weight_decay
        )
        optimizer = torch.optim.AdamW(param_groups, **optimizer_config.kwargs.__dict__)
    else:
        raise NotImplementedError

    scheduler_config = config.scheduler
    if scheduler_config.type == "CosLR":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=scheduler_config.kwargs.epochs,
            lr_min=1e-6,
            k_decay=0.1,
            warmup_lr_init=1e-6,
            warmup_t=scheduler_config.kwargs.initial_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )
    elif scheduler_config.type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **scheduler_config.kwargs
        )
    elif scheduler_config.type == "LambdaLR":
        scheduler = build_lambda_sche(optimizer, scheduler_config.kwargs)  # misc.py
    elif scheduler_config.type == "function":
        scheduler = None
    else:
        raise NotImplementedError()

    if hasattr(config, "bnmscheduler"):
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == "Lambda":
            bnscheduler = build_lambda_bnsche(model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]

    return optimizer, scheduler