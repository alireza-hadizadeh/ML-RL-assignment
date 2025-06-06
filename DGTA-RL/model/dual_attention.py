import torch.nn as nn
import torch
import math

class DualAttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_head=8, dim_ff=256):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ta = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.linear = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, H):  # H: (B, L, D)
        Hs, _ = self.sa(H, H, H)
        Ht, _ = self.ta(H, H, H)
        H_concat = torch.cat([Hs, Ht], dim=-1)
        H_int = self.linear(H_concat)
        H = self.norm1(H + H_int)
        H = self.norm2(H + self.ff(H))
        return H
