# dgta_rl/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- helpers ----------
def clones(module, N):
    return nn.ModuleList([module for _ in range(N)])

class PositionalEncoding(nn.Module):
    """Standard transformer sinusoidal PE."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

# ---------- Dual Attention Block (spatial + temporal) ----------
class DualAttentionBlock(nn.Module):
    """
    Implements the dual attention layer described in the paper (Eq. 11–13).
    """
    def __init__(self, d_model=128, n_head=8, dim_ff=256):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ta = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.linear = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff),
                                nn.ReLU(),
                                nn.Linear(dim_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, H):  # H: (B, L, D)
        Hs, _ = self.sa(H, H, H)
        Ht, _ = self.ta(H, H, H)
        H_concat = torch.cat([Hs, Ht], dim=-1)
        H_int = self.linear(H_concat)
        H = self.norm1(H + H_int)
        H = self.norm2(H + self.ff(H))
        return H

# ---------- Dynamic Encoder ----------
class DynamicEncoder(nn.Module):
    """
    Selects representations corresponding to current node/time and fuses realised travel times.
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, H, current_idx, t_idx):
        B, L, D = H.shape
        T = t_idx.max().item() + 1
        N = L // T

        # --- Extract representation at time t_idx ---
        t_base = t_idx * N  # (B,)
        t_idx_expanded = t_base.unsqueeze(1) + torch.arange(N, device=H.device)  # (B, N)
        t_idx_expanded = torch.clamp(t_idx_expanded, max=L - 1)
        H_t = torch.gather(H, 1, t_idx_expanded.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)

        # --- Extract representation at current node across time ---
        idx_flat = t_base + current_idx.squeeze(1)  # (B,)
        idx_flat = torch.clamp(idx_flat, max=L - 1)  # avoid overflow
        H_T = torch.gather(H, 1, idx_flat.unsqueeze(1).unsqueeze(-1).expand(-1, 1, D))  # (B, 1, D)
        H_T = H_T.expand(-1, N, -1)  # (B, N, D)

        fused = torch.tanh(self.gate(torch.cat([H_t, H_T], dim=-1)))  # (B, N, D)
        return fused


# ---------- Decoder with Pointer Mechanism ----------
class PointerDecoder(nn.Module):
    def __init__(self, d_model, n_head=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    def forward(self, H, mask):
        q = H[:, :1, :]  # (B,1,D)
        attn_out, _ = self.attn(q, H, H, key_padding_mask=mask)
        logits = (q @ H.transpose(1, 2)).squeeze(1) / math.sqrt(H.size(-1))
        logits = logits.masked_fill(mask, -1e9)
        return logits

# ---------- DGTA model ----------
class DGTA(nn.Module):
    """
    Full architecture = Embedding + L dual blocks + DynamicEncoder + PointerDecoder.
    """
    def __init__(self, d_model=128, n_head=8, L=3):
        super().__init__()
        self.coord_embed = nn.Linear(2, d_model // 2)
        self.time_embed = nn.Embedding(24, d_model // 2)
        self.pe = PositionalEncoding(d_model)
        self.blocks = clones(DualAttentionBlock(d_model, n_head), L)
        self.dynamic_enc = DynamicEncoder(d_model)
        self.decoder = PointerDecoder(d_model, n_head)

    def forward(self, coords, t_idx, visited_mask, current_idx):
        B, N, _ = coords.shape
        E = torch.cat([self.coord_embed(coords),
                       self.time_embed(t_idx).repeat(1, N, 1)], dim=-1)
        H = self.pe(E)

        T = t_idx.max().item() + 1
        H = H.repeat(1, T, 1)

        for blk in self.blocks:
            H = blk(H)

        H_dyn = self.dynamic_enc(H, current_idx, t_idx)
        logits = self.decoder(H_dyn, visited_mask)
        return logits
