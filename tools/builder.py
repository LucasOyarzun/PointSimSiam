import os
import torch


def resume_model(base_model, config):
    ckpt_path = os.path.join(config.experiment_path, "ckpt-last.pth")
    if not os.path.exists(ckpt_path):
        print(f"[RESUME INFO] no checkpoint file from path {ckpt_path}...")
        return 0, 0
    print(f"[RESUME INFO] Loading model weights from {ckpt_path}...")

    # load state dict
    map_location = {"cuda:%d" % 0: "cuda:%d" % config.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if config.local_rank == 0:
    base_ckpt = {
        k.replace("module.", ""): v for k, v in state_dict["base_model"].items()
    }
    base_model.load_state_dict(base_ckpt, strict=True)

    # parameter
    start_epoch = state_dict["epoch"] + 1
    best_metrics = state_dict["best_metrics"]
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print(
        f"[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})"
    )
    return start_epoch, best_metrics


def resume_optimizer(optimizer, config):
    ckpt_path = os.path.join(config.experiment_path, "ckpt-last.pth")
    if not os.path.exists(ckpt_path):
        print(f"[RESUME INFO] no checkpoint file from path {ckpt_path}...")
        return 0, 0, 0
    print(f"[RESUME INFO] Loading optimizer from {ckpt_path}...")
    # load state dict
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # optimizer
    optimizer.load_state_dict(state_dict["optimizer"])


def save_checkpoint(
    base_model, optimizer, epoch, metrics, best_metrics, prefix, config
):
    if config.local_rank == 0:
        torch.save(
            {
                "base_model": base_model.module.state_dict()
                if config.distributed
                else base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics.state_dict() if metrics is not None else dict(),
                "best_metrics": best_metrics.state_dict()
                if best_metrics is not None
                else dict(),
            },
            os.path.join(config.experiment_path, prefix + ".pth"),
        )
        print(
            f"Save checkpoint at {os.path.join(config.experiment_path, prefix + '.pth')}"
        )


def load_model(base_model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError("no checkpoint file from path %s..." % ckpt_path)
    print(f"Loading weights from {ckpt_path}...")

    # load state dict
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # parameter resume of base model
    if state_dict.get("model") is not None:
        base_ckpt = {
            k.replace("module.", ""): v for k, v in state_dict["model"].items()
        }
    elif state_dict.get("base_model") is not None:
        base_ckpt = {
            k.replace("module.", ""): v for k, v in state_dict["base_model"].items()
        }
    else:
        raise RuntimeError("mismatch of ckpt weight")
    base_model.load_state_dict(base_ckpt, strict=True)

    epoch = -1
    if state_dict.get("epoch") is not None:
        epoch = state_dict["epoch"]
    if state_dict.get("metrics") is not None:
        metrics = state_dict["metrics"]
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = "No Metrics"
    print(f"ckpts @ {epoch} epoch( performance = {str(metrics):s})")
    return
