import torch
import torch.nn as nn


class CNNBiLSTMAttention(nn.Module):
    def __init__(
        self,
        num_classes: int = 16,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
    ):
        super().__init__()

        # CNN feature extractor
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

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=256 * 8,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )

        # Attention scoring
        # Each BiLSTM timestep gets a learned importance score.
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x):
        # x: [batch, 1, 128, 1292]
        x = self.cnn(x)

        # After CNN:
        # [batch, 256, 8, 80]
        batch, channels, height, width = x.shape

        # Convert frequency dimension into the feature dimension.
        # Temporal dimension becomes sequence length.
        x = x.permute(0, 3, 1, 2)

        # [batch, 80, 256 * 8]
        x = x.reshape(batch, width, channels * height)

        # BiLSTM
        # [batch, 80, 256]
        x, _ = self.lstm(x)

        # Attention scores
        # [batch, 80, 1]
        attention_scores = self.attention(x)

        # Normalize across time.
        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        # Weighted temporal pooling
        # [batch, 1, 80] @ [batch, 80, 256]
        # -> [batch, 1, 256]
        x = torch.bmm(
            attention_weights.transpose(1, 2),
            x,
        )

        # [batch, 256]
        x = x.squeeze(1)

        # Classification
        x = self.classifier(x)

        return x