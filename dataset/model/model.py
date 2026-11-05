import torch
import torch.nn as nn

class LandmarkTransformer(nn.Module):
    def __init__(self, input_dim=63, embed_dim=64, num_heads=4, num_layers=2, num_classes=5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.embedding(x).unsqueeze(1)   # Add seq dim [batch, seq=1, embed]
        x = self.transformer(x)              # [batch, seq, embed]
        x = x[:, 0, :]                       # Take first token
        x = self.fc(x)
        return x
