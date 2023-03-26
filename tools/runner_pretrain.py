import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import LinearSVC
from models import get_model
from optimizers import get_optimizer_sche
from datasets import ShapeNet55Dataset
from tools import builder


class Acc_Metric:
    """Metric for accuracy"""

    def __init__(self, acc=0.0):
        if type(acc).__name__ == "dict":
            self.acc = acc["acc"]
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict["acc"] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    """SVM evaluation"""
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1.0 / pred.shape[0]


def run_net(config):
    # Dataset
    train_dataset = ShapeNet55Dataset(config=config.dataset.train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=config.dataset.train.others.batch_size
    )

    # Model
    model = get_model(config=config.model)
    if config.use_gpu:
        model.to(config.local_rank)

    # Parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.0)
    metrics = Acc_Metric(0.0)

    # resume ckpts
    if config.resume:
        start_epoch, best_metric = builder.resume_model(model, config)
        best_metrics = Acc_Metric(best_metric)
    elif config.start_ckpts is not None:
        builder.load_model(model, config.start_ckpts)

    # DDP
    if config.distributed:
        # Sync BN
        if config.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print("Using Synchronized BatchNorm ...")
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print("Using Distributed Data parallel ...")
    else:
        print("Using Data parallel ...")
        config.device = torch.device("cuda" if config.use_gpu else "cpu")
        model = nn.DataParallel(model).cuda()

    # Optimizer and Scheduler
    optimizer, scheduler = get_optimizer_sche(model, config)

    if config.resume:
        builder.resume_optimizer(optimizer, config)

    # Train
    model.zero_grad()
    for epoch in range(config.epochs):
        model.train()

        for index, (tax, obj_id, data1, data2, data3) in enumerate(train_loader):
            model.zero_grad()
            if not config.distributed:
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
