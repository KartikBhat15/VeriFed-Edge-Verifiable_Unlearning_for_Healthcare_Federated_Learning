"""
Compact 1D CNN for 12-lead ECG classification.
Input: (batch, 12, T) where T=1000 for PTB-XL at 100Hz over 10s.
Output: (batch, 5) logits for NORM, MI, STTC, CD, HYP (multi-label -> sigmoid at inference).
"""

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    def __init__(self, in_leads: int = 12, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model_params(model: nn.Module):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_params(model: nn.Module, params):
    import numpy as np
    state_dict = model.state_dict()
    for key, val in zip(state_dict.keys(), params):
        state_dict[key] = torch.tensor(np.array(val))
    model.load_state_dict(state_dict, strict=True)
