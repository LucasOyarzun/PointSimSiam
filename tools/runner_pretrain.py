import torch
import os
from models import get_model
from optimizers import get_optimizer
from datasets import ShapeNet55Dataset
from timm.scheduler import CosineLRScheduler


def run_net(config):
    train_dataset = ShapeNet55Dataset(config=config.dataset.train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=config.dataset.train.others.batch_size
    )

    model = get_model(config=config.model)

    optimizer = get_optimizer(model, config.optimizer)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=config.scheduler.kwargs.epochs,
        lr_min=1e-6,
        k_decay=0.1,
        warmup_lr_init=1e-6,
        warmup_t=config.scheduler.kwargs.initial_epochs,
        cycle_limit=1,
        t_in_epochs=True,
    )

    for epoch in range(config.epochs):
        model.train()

        for index, (tax, obj_id, data1, data2, data3) in enumerate(train_loader):
            model.zero_grad()
            data1 = data1.to(config.device, non_blocking=True)
            data2 = data2.to(config.device, non_blocking=True)

            data1 = data1.transpose(2, 1).contiguous()
            data2 = data2.transpose(2, 1).contiguous()

            data_dict = model(data1, data2)
            loss = data_dict["loss"].mean()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            print(f"{index}/{len(train_loader)} - Loss: {loss.item()}")

        model_path = os.path.join(config.ckpt_dir, f"{config.name}_{epoch+1}")
        torch.save(
            {"epoch": epoch + 1, "state_dict": model.module.state_dict()}, model_path
        )

        print(f"Model saved to {model_path}")
