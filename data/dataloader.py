# data/dataloader.py

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, download=True, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_set = MNIST(root=data_dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
