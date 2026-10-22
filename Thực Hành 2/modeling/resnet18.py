import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.bn_2(self.conv_2(x))
        return self.relu(x + residual)


class ResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        self.block_1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.block_2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0), ResidualBlock(64, 128), ResidualBlock(128, 128))
        self.block_3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0), ResidualBlock(128, 256), ResidualBlock(256, 256))
        self.block_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0), ResidualBlock(256, 512), ResidualBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(512, num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return self.output(torch.flatten(self.pool(x), start_dim=1))
