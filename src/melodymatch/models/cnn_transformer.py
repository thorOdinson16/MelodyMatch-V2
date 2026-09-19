import torch
import torch.nn as nn


class CNNTransformer(nn.Module):

    def __init__(
        self,
        num_classes: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # CNN output:
        # [B, 256, 8, ~80]
        #
        # Each time step contains:
        # 256 channels × 8 frequency bins = 2048 features.

        self.projection = nn.Linear(
            256 * 8,
            d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                81,
                d_model,
            )
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                d_model,
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

    def forward(self, x):

        # ----------------------------------------------------
        # CNN feature extraction
        # ----------------------------------------------------

        x = self.cnn(x)

        # [B, C, H, W]
        batch, channels, height, width = x.shape

        # ----------------------------------------------------
        # Convert CNN feature map into temporal tokens
        # ----------------------------------------------------

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        # [B, W, C*H]
        x = x.reshape(
            batch,
            width,
            channels * height,
        )

        # ----------------------------------------------------
        # Project CNN features into Transformer dimension
        # ----------------------------------------------------

        x = self.projection(x)

        # ----------------------------------------------------
        # Add CLS token
        # ----------------------------------------------------

        cls_token = self.cls_token.expand(
            batch,
            -1,
            -1,
        )

        x = torch.cat(
            [cls_token, x],
            dim=1,
        )

        # ----------------------------------------------------
        # Positional embeddings
        # ----------------------------------------------------

        x = x + self.positional_embedding[
            :, :x.size(1), :
        ]

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.transformer(x)

        # CLS representation
        x = self.norm(x[:, 0])

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.classifier(x)

        return x