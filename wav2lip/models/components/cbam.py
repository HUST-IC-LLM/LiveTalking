import torch
from torch import nn


class CBAMChannelAttn(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.shared_mlp(self.avg_pool(x))
        max = self.shared_mlp(self.max_pool(x))
        return x * self.act(avg + max)


class CBAMSpatialAttn(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max], dim=1)
        return x * self.act(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = CBAMChannelAttn(in_planes, ratio)
        self.sa = CBAMSpatialAttn(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        return self.sa(x)
