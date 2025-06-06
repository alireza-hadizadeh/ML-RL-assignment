import torch.nn as nn
import torch
import math

class PointerDecoder(nn.Module):
    def __init__(self, d_model, n_head=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    def forward(self, H, mask):
        q = H[:, :1, :]
        attn_out, _ = self.attn(q, H, H, key_padding_mask=mask)
        logits = (q @ H.transpose(1, 2)).squeeze(1) / math.sqrt(H.size(-1))
        logits = logits.masked_fill(mask, -1e9)
        return logits
