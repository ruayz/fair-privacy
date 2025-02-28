import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, in_shape, num_classes=10):
        super().__init__()
        if in_shape == 3:
            # assume CIFAR-10
            in_shape = 3072
        elif in_shape == 1:
            # assume mnist
            in_shape = 784
        elif type(in_shape) == int:
            # assume tabular
            in_shape = in_shape
        self.fc1 = nn.Linear(in_shape, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        inputs = torch.tanh(self.fc1(inputs))
        outputs = self.fc2(inputs)
        return outputs


# class MLP(nn.Module):
#     """For Texas100 dataset."""

#     def __init__(self, in_shape, num_classes=2, activation=nn.Tanh):
#         super().__init__()
#         layers = []
#         n_units_list = [in_shape, 2048, 1024, 512, 256, num_classes]
#         prev_layer_size = n_units_list[0]
#         for n_units in n_units_list[1:-1]:
#             layers.append(nn.Linear(in_features=prev_layer_size, out_features=n_units))
#             prev_layer_size = n_units
#             layers.append(activation())
#         layers.append(nn.Linear(in_features=prev_layer_size, out_features=n_units_list[-1]))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)
