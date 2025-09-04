from typing import List

import torch
from torch import nn

from .components import CBAM, LGCM
from .utils import Conv2d, Conv2dTranspose


class FaceDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=1, padding=0
                    ),  # 3,3
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2dTranspose(
                        1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 6, 6
                nn.Sequential(
                    Conv2dTranspose(
                        768, 384, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 12, 12
                nn.Sequential(
                    Conv2dTranspose(
                        512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 24, 24
                nn.Sequential(
                    Conv2dTranspose(
                        320, 128, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),  # 48, 48
                nn.Sequential(
                    Conv2dTranspose(
                        160, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
            ]
        )  # 96,96

    def forward(
        self, audio_embedding: torch.Tensor, feats: List[torch.Tensor]
    ) -> torch.Tensor:
        x = audio_embedding
        for f in self.blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()
        return x


class Block_v2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mul3: bool = False):
        super().__init__()
        self.up = Conv2dTranspose(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2 if not mul3 else 1,
            padding=1 if not mul3 else 0,
            output_padding=1 if not mul3 else 0,
        )
        self.res = Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            residual=True,
        )
        self.cbam = CBAM(out_ch)
        self.lgcm = LGCM(out_ch, 8)

    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        x = self.cbam(x)
        x = self.lgcm(x)
        return x


class FaceDecoder_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.ModuleList(
            [
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                Block_v2(1024, 512, mul3=True),
                Block_v2(1024, 512),
                Block_v2(768, 384),
                Block_v2(512, 256),
                Block_v2(320, 128),
                Block_v2(160, 64),
            ]
        )

    def forward(
        self, audio_embedding: torch.Tensor, feats: List[torch.Tensor]
    ) -> torch.Tensor:
        x = audio_embedding
        for f in self.block:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            feats.pop()
        return x
