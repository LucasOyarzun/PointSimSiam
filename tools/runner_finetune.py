import os
import torch
import torch.nn as nn
from models import get_model
from optimizers import get_optimizer_sche
from datasets import ModelNetDataset


def run_net(config):
    # Dataset
    dataset_config = config.dataset.train
    train_dataset = ModelNetDataset(config=dataset_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.total_bs,
                                                shuffle = dataset_config.others.subset == 'train', 
                                                drop_last = dataset_config.others.subset == 'train',
                                                num_workers = int(config.num_workers))
    config.device = torch.device("cuda" if config.use_gpu else "cpu")
    # Model
    model = get_model(config=config.model)
    # load pretrained weights
    checkpoint_path = os.path.join(config.ckpt_dir, f"ShapeNet_1.pth")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['state_dict'])
    # if 'module' in list(checkpoint['state_dict'].keys())[0]:
    #     # Eliminar el prefijo "module." de las claves del modelo
    #     state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    #     # Crear una instancia del modelo
    #     model = get_model(config=config.model)
    #     # Cargar los parámetros del modelo
    #     model.load_state_dict(state_dict)
    model.module.predictor = nn.Sequential(
                nn.Linear(384 * 2, 256), # trans_dim = 384 en point MAE
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 40) # cls_dim = 40 en point MAE
            )
    model.module.forward = model.module.forward_finetune
    model.cuda()

    # Optimizer and Scheduler
    optimizer, scheduler = get_optimizer_sche(model, config)
   

    # Entrena el modelo con tus datos de entrenamiento
    for epoch in range(config.epochs):
        running_loss = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0
        npoints = config.npoints
        for index, (model_name, text_sample, (current_points, label)) in enumerate(train_loader):
            current_points = current_points.cuda()
            label = label.cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if current_points.size(1) < point_all:
                point_all = current_points.size(1)

            optimizer.zero_grad()
            print(current_points.shape)
            preds = model(current_points)
            loss = nn.CrossEntropyLoss(preds, label)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            # calcular el número de predicciones correctas y actualizar el contador
            _, predicted = torch.max(preds, 1)
            correct_predictions += (predicted == label).sum().item()
            total_predictions += label.size(0)
             # imprimir el loss y la precisión actual
            running_loss += loss.item() * current_points.size(0)
            print("Loss: ", loss.item())
            # if (i + 1) % 2 == 0:
            #     avg_loss = running_loss / 2
            #     accuracy = correct_predictions / total_predictions
            #     print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            #     running_loss = 0.0
            #     correct_predictions = 0.0
            #     total_predictions = 0.0

