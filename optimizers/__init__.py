import torch


def get_optimizer(model, config):
    predictor_prefix = ("module.predictor", "predictor")
    parameters = [
        {
            "name": "base",
            "params": [
                param
                for name, param in model.named_parameters()
                if not name.startswith(predictor_prefix)
            ],
            "lr": config.kwargs.lr,
        },
        {
            "name": "predictor",
            "params": [
                param
                for name, param in model.named_parameters()
                if name.startswith(predictor_prefix)
            ],
            "lr": config.kwargs.lr,
        },
    ]

    if config.name == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            **config.kwargs.__dict__,
            nesterov=True,
        )
    elif config.name == "Adam":
        optimizer = torch.optim.Adam(parameters, **config.kwargs.__dict__)
    elif config.name == "AdamW":

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

        param_groups = add_weight_decay(model, weight_decay=config.kwargs.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, **config.kwargs.__dict__)
    else:
        raise NotImplementedError

    return optimizer
