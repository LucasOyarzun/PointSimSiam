import torch
import random
import os
from models import get_model
from optimizers import get_optimizer, LR_Scheduler
from datasets import ShapeNet55Dataset
from utils.parser import parse_args


def main():
    # Parse arguments (console + config file)
    config = parse_args()

    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ShapeNet55Dataset(config=config, npoints=1024, split="train")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=config.train.batch_size
    )

    model = get_model(config=config.model)

    optimizer = get_optimizer(
        config.train.optimizer.name,
        model,
        lr=config.train.base_lr * config.train.batch_size / 256,
        momentum=config.train.optimizer.momentum,
        weight_decay=config.train.optimizer.weight_decay,
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        config.train.warmup_epochs,
        config.train.warmup_lr * config.train.batch_size / 256,
        config.train.num_epochs,
        config.train.base_lr * config.train.batch_size / 256,
        config.train.final_lr * config.train.batch_size / 256,
        len(train_loader),
        constant_predictor_lr=True,  # see the end of section 4.2 predictor
    )

    for epoch in range(config.train.stop_at_epoch):
        model.train()

        for index, (tax, obj_id, data1, data2, data3) in enumerate(train_loader):
            model.zero_grad()
            data1 = data1.to(device, non_blocking=True)
            data2 = data2.to(device, non_blocking=True)

            data1 = data1.transpose(2, 1).contiguous()
            data2 = data2.transpose(2, 1).contiguous()

            data_dict = model(data1, data2)
            loss = data_dict["loss"].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            print(f"{index}/{len(train_loader)} - Loss: {loss.item()}")

        model_path = os.path.join(config.ckpt_dir, f"{config.name}_{epoch+1}")
        torch.save(
            {"epoch": epoch + 1, "state_dict": model.module.state_dict()}, model_path
        )

        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
