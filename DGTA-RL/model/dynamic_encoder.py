import torch.nn as nn
import torch

class DynamicEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, H, current_idx, t_idx):
        B, L, D = H.shape
        T = t_idx.max().item() + 1
        N = L // T

        t_base = t_idx * N
        t_idx_expanded = t_base.unsqueeze(1) + torch.arange(N, device=H.device)
        t_idx_expanded = torch.clamp(t_idx_expanded, max=L - 1)
        H_t = torch.gather(H, 1, t_idx_expanded.unsqueeze(-1).expand(-1, -1, D))

        idx_flat = t_base + current_idx.squeeze(1)
        idx_flat = torch.clamp(idx_flat, max=L - 1)
        H_T = torch.gather(H, 1, idx_flat.unsqueeze(1).unsqueeze(-1).expand(-1, 1, D))
        H_T = H_T.expand(-1, N, -1)

        fused = torch.tanh(self.gate(torch.cat([H_t, H_T], dim=-1)))
        return fused
