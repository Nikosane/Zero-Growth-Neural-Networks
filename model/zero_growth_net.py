# model/zero_growth_net.py

import torch
import torch.nn as nn
from model.layers import SparseLinear

class ZeroGrowthNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sparsity):
        super().__init__()
        self.fc1 = SparseLinear(input_size, hidden_size, sparsity)
        self.relu = nn.ReLU()
        self.fc2 = SparseLinear(hidden_size, output_size, sparsity)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def rewire(self, strategy="lottery_ticket"):
        self.fc1.rewire(strategy)
        self.fc2.rewire(strategy)
