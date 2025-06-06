# dgta_rl/model/dgta.py
import torch
import torch.nn as nn
from model.positional import PositionalEncoding
from model.dual_attention import DualAttentionBlock
from model.dynamic_encoder import DynamicEncoder
from model.pointer_decoder import PointerDecoder
from utils import clones

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
