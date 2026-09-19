import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):

    def __init__(
        self,
        num_classes: int = 16,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
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

        # After four 2x2 pooling operations:
        # [B, 256, 8, ~80]
        #
        # Treat the time dimension as the sequence:
        # [B, ~80, 256*8]

        self.lstm = nn.LSTM(
            input_size=256 * 8,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                lstm_hidden * 2,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.cnn(x)

        # [B, C, H, W]
        batch, channels, height, width = x.shape

        # W = temporal sequence
        # Move W before C/H:
        x = x.permute(
            0, 3, 1, 2
        )

        # [B, W, C*H]
        x = x.reshape(
            batch,
            width,
            channels * height,
        )

        x, _ = self.lstm(x)

        # Mean pooling over temporal sequence
        x = x.mean(dim=1)

        x = self.classifier(x)

        return x