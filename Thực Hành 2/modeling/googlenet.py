import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.branch_1 = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=1), nn.ReLU())
        self.branch_2 = nn.Sequential(nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(), nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU())
        self.branch_3 = nn.Sequential(nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(), nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU())
        self.branch_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True), nn.Conv2d(in_channels, c4, kernel_size=1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.block_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(), nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.block_3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32), Inception(256, 128, (128, 192), (32, 96), 64), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.block_4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64), Inception(512, 160, (112, 224), (24, 64), 64), Inception(512, 128, (128, 256), (24, 64), 64), Inception(512, 112, (144, 288), (32, 64), 64), Inception(528, 256, (160, 320), (32, 128), 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.block_5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128), Inception(832, 384, (192, 384), (48, 128), 128), nn.AdaptiveAvgPool2d((1, 1)))
        self.output = nn.Linear(1024, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return self.output(torch.flatten(x, start_dim=1))
