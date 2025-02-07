import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from numpy.random import randint


class ShapeNet55Dataset(data.Dataset):
    def __init__(self, config, npoints, split):
        self.data_root = os.path.join(
            os.path.abspath(os.getcwd()), "data\\ShapeNet55-34\\ShapeNet-55"
        )
        self.pc_path = os.path.join(
            os.path.abspath(os.getcwd()), "data\\ShapeNet55-34\\shapenet_pc"
        )
        self.subset = split
        self.npoints = npoints
        self.config = config

        self.train_data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        self.test_data_list_file = os.path.join(self.data_root, "test.txt")

        self.sample_points_num = config.npoints

        with open(self.train_data_list_file, "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split("-")[0]
            model_id = line.split("-")[1].split(".")[0]
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line}
            )

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        pc = pc[np.random.choice(pc.shape[0], num, replace=False)]
        return pc

    def make_holes_pcd(self, pcd, num_holes, augmentation_size=0.1):
        """
        Makes holes in the point cloud data

        Parameters
        ----------
        pcd : float[n,3]
            point cloud data of n size in x, y, z format
        num_holes : int
            number of holes to make
        augmentation_size : float
            size of the holes normalized, by default 0.1

        Returns
        -------
        float[n,3]
            point cloud data in x, y, z of n size format
        """
        new_pcd = pcd
        for i in range(num_holes):
            rand_point = new_pcd[
                randint(0, new_pcd.shape[0])
            ]  # Pick a random hole center
            partial_pcd = []
            for i in range(new_pcd.shape[0]):  # Check if the point is in the hole
                dist = np.linalg.norm(rand_point - new_pcd[i])
                if (
                    dist >= augmentation_size
                ):  # If not, add it to the partial point cloud
                    partial_pcd = partial_pcd + [new_pcd[i]]
            new_pcd = np.array([np.array(e) for e in partial_pcd])
        return new_pcd

    def add_noise_pcd(self, pcd, augmentation_size=0.1):
        """
        Adds noise to the point cloud data
        """
        noise = np.random.normal(0, augmentation_size, pcd.shape)
        return pcd + noise

    def resample_pcd(self, pcd, n):
        """
        Drop or duplicate points so that pcd has exactly n points
        """
        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate(
                [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])]
            )
        return pcd[idx[:n]]

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample["file_path"])).astype(
            np.float32
        )

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)

        if self.config.dataset.augmentation == "holes":
            data1 = self.resample_pcd(
                self.make_holes_pcd(
                    data,
                    num_holes=self.config.dataset.num_holes,
                    augmentation_size=self.config.dataset.augmentation_size,
                ),
                self.sample_points_num,
            )
            data2 = self.resample_pcd(
                self.make_holes_pcd(
                    data,
                    num_holes=self.config.dataset.num_holes,
                    augmentation_size=self.config.dataset.augmentation_size,
                ),
                self.sample_points_num,
            )
        elif self.config.dataset.augmentation == "noise":
            data1 = self.resample_pcd(
                self.add_noise_pcd(
                    data,
                    augmentation_size=self.config.dataset.augmentation_size,
                ),
                self.sample_points_num,
            )
            data2 = self.resample_pcd(
                self.add_noise_pcd(
                    data,
                    augmentation_size=self.config.dataset.augmentation_size,
                ),
                self.sample_points_num,
            )

        data1 = torch.from_numpy(data1).float()
        data2 = torch.from_numpy(data2).float()
        return sample["taxonomy_id"], sample["model_id"], data1, data2, data

    def __len__(self):
        return len(self.file_list)
