"""Neural network architecture for Kickstarter Success Predictor.

Extracted from kickstarterModel.py so both training scripts and the
FastAPI lifespan loader (Phase 2) can import a single canonical class.
"""
from __future__ import annotations

import torch.nn as nn


class KickstarterNet(nn.Module):
    """Fully-connected classifier matching the current kickstarterModel.py
    architecture: 4 linear layers with BatchNorm + ReLU + Dropout, final
    linear layer outputs a single logit (sigmoid applied externally via
    BCEWithLogitsLoss).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),           nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),           nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)
