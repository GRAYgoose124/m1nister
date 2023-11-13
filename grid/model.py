import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# define model
class ConvAttention(nn.Module):
    def __init__(self, in_channels):
        super(ConvAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        attention = torch.sigmoid(self.conv1(x))
        return x * attention


class ParallelPath(nn.Module):
    def __init__(self):
        super(ParallelPath, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.attention1 = ConvAttention(16)
        self.attention2 = ConvAttention(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention1(x)
        x = F.relu(self.pool(x))

        x = self.conv2(x)
        x = self.attention2(x)
        x = F.relu(self.pool(x))

        x = torch.flatten(x, 1)
        return x


class GridNetwork(nn.Module):
    def __init__(self):
        super(GridNetwork, self).__init__()
        self.path1 = ParallelPath()
        self.path2 = ParallelPath()

        self.fc1 = nn.Linear(32 * 5 * 5 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)

        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
