

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """Optimized SwiGLU with fused gate and up projection."""
    def __init__(self, d_model, dropout=0.0, hidden_dim_multiplier: float = 2/3):
        super().__init__()
        
        # LLaMA style hidden dim calculation
        hidden_dim = int(hidden_dim_multiplier * 4 * d_model)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        # Optimization: Merge w1 (Gate) and w3 (Up) into one Linear layer
        # Output dim is 2 * hidden_dim
        self.w_gate_up = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w2._is_scaled_std_output = True 
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Fused projection -> Chunk
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        
        # SwiGLU Logic: (SiLU(Gate) * Up)
        x = F.silu(gate) * up
        
        # Down projection
        x = self.w2(x)
        x = self.dropout(x)
        
        return x
