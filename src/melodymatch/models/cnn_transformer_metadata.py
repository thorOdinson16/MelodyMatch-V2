import torch
import torch.nn as nn


class CNNTransformerMetadata(nn.Module):

    def __init__(
        self,
        metadata_dim,
        num_classes=16,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Audio CNN
        # ----------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.projection = nn.Linear(
            256 * 8,
            d_model,
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.positional_embedding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    81,
                    d_model,
                )
            )
        )

        self.audio_norm = nn.LayerNorm(
            d_model
        )

        # ----------------------------------------------------
        # Metadata branch
        # ----------------------------------------------------

        self.metadata_mlp = nn.Sequential(
            nn.Linear(
                metadata_dim,
                128,
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(
                128,
                128,
            ),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                d_model + 128,
                256,
            ),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes,
            ),
        )

        nn.init.normal_(
            self.cls_token,
            std=0.02,
        )

        nn.init.normal_(
            self.positional_embedding,
            std=0.02,
        )

    def forward(
        self,
        mel,
        metadata,
    ):

        # ----------------------------------------------------
        # Audio branch
        # ----------------------------------------------------

        x = self.cnn(mel)

        batch, channels, height, width = (
            x.shape
        )

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        x = x.reshape(
            batch,
            width,
            channels * height,
        )

        x = self.projection(x)

        cls_token = (
            self.cls_token.expand(
                batch,
                -1,
                -1,
            )
        )

        x = torch.cat(
            [cls_token, x],
            dim=1,
        )

        x = (
            x
            + self.positional_embedding[
                :, :x.size(1), :
            ]
        )

        x = self.transformer(x)

        audio_embedding = (
            self.audio_norm(
                x[:, 0]
            )
        )

        # ----------------------------------------------------
        # Metadata branch
        # ----------------------------------------------------

        metadata_embedding = (
            self.metadata_mlp(
                metadata
            )
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fused = torch.cat(
            [
                audio_embedding,
                metadata_embedding,
            ],
            dim=1,
        )

        return self.classifier(
            fused
        )