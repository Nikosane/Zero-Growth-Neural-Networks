# model/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.sparsity = sparsity
        self.mask = self.init_mask()

    def init_mask(self):
        total = self.in_features * self.out_features
        k = int((1 - self.sparsity) * total)
        mask = torch.zeros(total, dtype=torch.bool)
        mask[:k] = 1
        mask = mask[torch.randperm(total)]
        return mask.view(self.out_features, self.in_features)

    def forward(self, x):
        w = self.weight * self.mask.float()
        return F.linear(x, w, self.bias)

    def rewire(self, strategy="lottery_ticket"):
        with torch.no_grad():
            weight_scores = torch.abs(self.weight)
            new_mask = self.mask.clone()

            if strategy == "lottery_ticket":
                flat_scores = weight_scores.view(-1)
                k = int((1 - self.sparsity) * flat_scores.numel())
                threshold, _ = torch.kthvalue(flat_scores, flat_scores.numel() - k + 1)
                new_mask = (weight_scores >= threshold).float()

            self.mask = new_mask.bool()
