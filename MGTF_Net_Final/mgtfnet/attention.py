"""CBAM and SE attention modules."""
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, reduced_planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_planes, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduced = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        z = self.gap(x).view(b, c)
        scale = self.fc(z).view(b, c, 1, 1)
        return x * scale
