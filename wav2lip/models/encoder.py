from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .attns import MulHeadCrossAttn
from .utils import Conv2d


class FaceEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(6, 16, kernel_size=7, stride=1, padding=3)
                ),  # 96,96
                nn.Sequential(
                    Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                    Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
            ]
        )

        self.fusions = nn.ModuleList(
            [
                MulHeadCrossAttn(32, 512, 8, d_model=128),
                MulHeadCrossAttn(128, 512, 8, d_model=256),
                MulHeadCrossAttn(512, 512, 8, d_model=512),
            ]
        )

    def forward(
        self, face_sequences: torch.Tensor, audio_embedding: torch.Tensor
    ) -> List[torch.Tensor]:
        input_dim = face_sequences.ndim
        if input_dim == 4:
            B, C, H, W = face_sequences.shape
            assert C == 6, "Face input should have 6 channels"

        elif input_dim == 5:
            B, C, T, H, W = face_sequences.shape
            assert C == 6, "Face input should have 6 channels"
            face_sequences = face_sequences.transpose(1, 2).view(B * T, C, H, W)

        else:
            raise ValueError("[ERROR] Face input should be 4D or 5D tensor")

        feats = []
        for i, block in enumerate(self.blocks):
            face_sequences = block(face_sequences)
            B, C, H, W = face_sequences.shape
            if i % 2 == 1:
                face_encoding = face_sequences.permute(0, 2, 3, 1).view(B, H * W, C)
                face_encoding = F.layer_norm(face_encoding, (C,))
                face_encoding = self.fusions[i // 2](face_encoding, audio_embedding)
                face_sequences += face_encoding.view(B, H, W, C).permute(0, 3, 1, 2)
            feats.append(face_sequences)

        return feats


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self, audio_sequences: torch.Tensor, a_alpha: float = 1.0
    ) -> torch.Tensor:
        if audio_sequences.ndim > 4:
            B, T, C, F, D = audio_sequences.shape
            assert C == 1, "Audio input should have a single channel"

            audio_sequences = audio_sequences.reshape(B * T, C, F, D)
            B = B * T

        else:
            B, C, F, D = audio_sequences.shape

        audio_embedding = self.blocks(audio_sequences)
        if a_alpha != 1.0:
            audio_embedding *= a_alpha
        return audio_embedding.view(B, 1, 512)

    def encode(self, audio_sequences: torch.Tensor) -> torch.Tensor:
        return self.forward(audio_sequences, a_alpha=1.0).transpose(1, 2).unsqueeze(-1)
