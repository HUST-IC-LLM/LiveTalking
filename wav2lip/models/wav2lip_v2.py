import torch
from torch import nn

from .decoder import FaceDecoder, FaceDecoder_v2
from .encoder import AudioEncoder, FaceEncoder
from .utils import Conv2d, nonorm_Conv2d


class Wav2Lip_v2(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.device = device

        self.face_encoder = FaceEncoder()
        self.audio_encoder = AudioEncoder()

        self.face_decoder = FaceDecoder_v2()

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio_sequences: torch.Tensor,
        face_sequences: torch.Tensor,
        a_alpha: float = 1.0,
    ) -> torch.Tensor:
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)
        input_dim_size = audio_sequences.ndim
        audio_embedding = self.audio_encoder(audio_sequences, a_alpha)

        feats = []
        feats = self.face_encoder(face_sequences, audio_embedding)

        x = audio_embedding.view(B, 512, 1, 1)
        x = self.face_decoder(x, feats)

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

    def audio_forward(
        self, audio_sequences: torch.Tensor, a_alpha: float = 1.0
    ) -> torch.Tensor:
        return self.audio_encoder(audio_sequences, a_alpha)

    def inference(
        self,
        audio_embedding: torch.Tensor,
        face_sequences: torch.Tensor,
    ) -> torch.Tensor:
        feats = self.face_encoder(face_sequences, audio_embedding)

        x = audio_embedding.view(audio_embedding.size(0), 512, 1, 1)
        x = self.face_decoder(x, feats)

        x = self.output_block(x)
        outputs = x

        return outputs


class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
                ),  # 48,96
                nn.Sequential(
                    nonorm_Conv2d(
                        32, 64, kernel_size=5, stride=(1, 2), padding=2
                    ),  # 48,48
                    nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                ),
                nn.Sequential(
                    nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24,24
                    nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
                ),
                nn.Sequential(
                    nonorm_Conv2d(
                        128, 256, kernel_size=5, stride=2, padding=2
                    ),  # 12,12
                    nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                ),
                nn.Sequential(
                    nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6,6
                    nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                    nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                    nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
            ]
        )

        self.binary_pred = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid()
        )
        self.label_noise = 0.0

    def get_lower_half(self, face_sequences: torch.Tensor) -> torch.Tensor:
        return face_sequences[:, :, face_sequences.size(2) // 2 :]

    def to_2d(self, face_sequences: torch.Tensor) -> torch.Tensor:
        B = face_sequences.size(0)
        face_sequences = torch.cat(
            [face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0
        )
        return face_sequences

    def perceptual_forward(self, false_face_sequences: torch.Tensor) -> torch.Tensor:
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(
            self.binary_pred(false_feats).view(len(false_feats), -1),
            torch.ones((len(false_feats), 1)).cuda(),
        )

        return false_pred_loss

    def forward(self, face_sequences: torch.Tensor) -> torch.Tensor:
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
