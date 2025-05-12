# utils/metrics.py

import torch

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).float().sum()
    return (correct / labels.shape[0]) * 100
