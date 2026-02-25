# model/lstm/model.py

import torch
import torch.nn as nn


class FallLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.classifier(out)
        return logits  # raw logits