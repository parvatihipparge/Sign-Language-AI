import torch
import torch.nn as nn


class LandmarkTransformer(nn.Module):

    def __init__(
        self,
        input_dim=126,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=37
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(0.3),

            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)