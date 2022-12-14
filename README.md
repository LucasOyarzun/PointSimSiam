# PointSimSiam

## Self supervised learning for 3D representations

In this work...


## 1. Requirements

PyTorch >= 1.7.0 < 1.11.0; python >= 3.7; CUDA >= 9.0; GCC >= 4.9; torchvision;

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## Point Simsiam Models

## Point Simsiam Pre-training

## Point Simsiam Fine-tuning

## Visualization

## Acknowledgements

Our codes are built upon [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [Simsiam-pytorch](https://github.com/PatrickHua/SimSiam)


## Reference