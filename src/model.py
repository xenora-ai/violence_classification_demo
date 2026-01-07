# src/model.py
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=1,
        bidirectional=False,
        freeze_cnn=True,
        unfreeze_last_k=0,
        temporal_pooling=False
    ):
        super().__init__()

        # Завантажуємо MobileNetV2 як backbone
        backbone = models.mobilenet_v2(weights=None)  # weights=None, бо ми завантажимо свої
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.temporal_pooling = temporal_pooling
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_out_dim, 1)
        )

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Згортаємо Batch та Time для CNN
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(B, T, -1)  # Повертаємо до [B, T, 1280]

        lstm_out, (h_n, _) = self.lstm(x)

        if self.temporal_pooling:
            feat = lstm_out.mean(dim=1)
        else:
            feat = h_n[-1]

        logits = self.classifier(feat)
        return logits
