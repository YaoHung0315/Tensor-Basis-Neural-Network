# network.py
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, width, layer, num_lambda, num_t):
        super(Net, self).__init__()
        # Create layers: input layer, hidden layers, and output layer
        layers = [nn.Linear(num_lambda, width), nn.LeakyReLU(0.01)]
        for _ in range(1, layer - 1):
            layers += [nn.Linear(width, width), nn.LeakyReLU(0.01)]
        layers.append(nn.Linear(width, num_t))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)