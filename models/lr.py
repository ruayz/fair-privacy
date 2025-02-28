"""Logistic Regression"""

import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, in_shape=None, num_classes=10):
        out_dim = num_classes
        if in_shape == 3:
            # assume CIFAR-10
            in_dim = 3072
        elif in_shape == 1:
            # assume mnist
            in_dim = 784
        elif type(in_shape) == int:
            # assume tabular
            in_dim = in_shape
        # elif len(in_shape) == 2:
        #     # B x D
        #     in_dim = in_shape[1]
        # else:
        #     # B x C x H x W
        #     in_dim = in_shape[1] * in_shape[2] * in_shape[3]

        super(LR, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        out = self.linear(torch.flatten(x, start_dim=1))
        return out