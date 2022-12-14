import h5py
import polyscope as ps
import numpy as np
import os, random, json, sys
from datasets import ShapeNet55Dataset, ScanObjectNNDataset
from arguments import parse_args


def visualize_point_cloud(args):
    ps.init()
    if args.visualization == "shapenet":
        # Load data
        dataset = ShapeNet55Dataset(config=args, npoints=1024, split="train")

        idx = random.randint(0, len(dataset))
        ps.register_point_cloud("Transform 1", dataset[idx][2], enabled=False)
        ps.register_point_cloud("Transform 2", dataset[idx][3], enabled=False)
        ps.register_point_cloud("Original", dataset[idx][4])

        # Print the class of the cloud
        with open("data\\ShapeNet55-34\\Shapenet_classes.json") as json_file:
            ShapeNET_classes = json.load(json_file)
            ps.info(f"Class: {ShapeNET_classes[dataset[idx][0]]}")

    if args.visualization == "scanobjectnn":
        dataset = ScanObjectNNDataset(config=args, split="train")
        idx = random.randint(0, len(dataset))

        ps.register_point_cloud("Point Cloud", dataset[idx][2][0])
        # Print the class of the cloud
        point_class = dataset[idx][2][1]
        with open("data\\ScanObjectNN\\scanobjectnn_classes.json") as json_file:
            ScanObjectNN_classes = json.load(json_file)["classes"]
            ps.info(f"Class: {ScanObjectNN_classes[point_class]}")

    ps.show()


if __name__ == "__main__":
    args = parse_args()
    visualize_point_cloud(args)
