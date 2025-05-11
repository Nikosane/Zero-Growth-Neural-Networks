# data/sample_dataset.py

"""
This file can be used to define a custom dataset if needed.
Currently, MNIST is used in dataloader.py.
"""

import torch
from torch.utils.data import Dataset

class SampleDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y
