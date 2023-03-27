import os
import torch
import torch.nn as nn
from models import get_model
from optimizers import get_optimizer_sche
from datasets import ShapeNet55Dataset

def run_net(config):
    # Dataset
    dataset_config = config.dataset.train
    train_dataset = ShapeNet55Dataset(config=dataset_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.total_bs,
                                                shuffle = dataset_config.others.subset == 'train', 
                                                drop_last = dataset_config.others.subset == 'train',
                                                num_workers = int(config.num_workers))

    # Model
    model = get_model(config=config.model)
    config.device = torch.device("cuda" if config.use_gpu else "cpu")
    model = nn.DataParallel(model).cuda()

    # Optimizer and Scheduler
    optimizer, scheduler = get_optimizer_sche(model, config)

    # Pretrain
    model.zero_grad()
    for epoch in range(config.epochs):
        model.train()
        for index, (tax, obj_id, data1, data2, data) in enumerate(train_loader):
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

        # Save checkpoint
        model_path = os.path.join(config.ckpt_dir, f"{config.name}_{epoch+1}")
        torch.save(
            {"epoch": epoch + 1, "state_dict": model.module.state_dict()}, model_path
        )
        print(f"Model saved to {model_path}")