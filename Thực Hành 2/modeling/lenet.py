import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.sigmoid(self.conv_1(x)))
        x = self.pool(torch.sigmoid(self.conv_2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc_1(x))
        x = torch.sigmoid(self.fc_2(x))
        return self.output(x)
