import numpy as np
import torch
from torch.utils.data import Dataset


class Lidar(Dataset):
    def __init__(self, distance, reflectivity, transform=None):
        self.distance = distance
        self.reflectivity = reflectivity
        self.len = len(distance)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            self.distance, self.reflectivity = self.transform(self.distance, self.reflectivity)

        self.distance = self.distance.view(1, 32, -1)
        self.reflectivity = self.reflectivity.view(1, 32, -1)

        return self.distance, self.reflectivity

    def __len__(self):
        return self.len
