import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from src.models.components.position_encoding import Rope1D, Rope2D
from src.models.components.ffn import SwiGLUFFN

class TransformerBlock(nn.Module):
    # __init__ 保持你原始的设计，无需任何修改
    def __init__(self, d_model, d_attn_scale=1, d_head=64, dropout=0.0, enable_cond_proj=False):
        super().__init__()
        
        d_attn = d_model * d_attn_scale
        self.d_head = d_head
        self.n_head = d_attn // self.d_head
        self.dropout = dropout

        self.rope1d = Rope1D(self.d_head)
        self.rope2d = Rope2D(self.d_head)
        
        # 共享的 QKV 投影和 Norm (x 和 h 共用)
        self.qkv_proj = nn.Linear(d_model, 3 * d_attn, bias=False)
        self.norm_in = nn.RMSNorm(d_model)

        self.norm_q = nn.RMSNorm(self.d_head)
        self.norm_k = nn.RMSNorm(self.d_head)
        
        self.alpha_hist_predictor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, self.n_head)
        )

        self.gate_attn_proj = nn.Sequential(
            nn.Linear(d_model, d_attn)
        )
        
        self.attn_out_proj = nn.Linear(d_attn, d_model, bias=False)
        self.attn_out_proj._is_scaled_std_output = True

        self.norm_ffn = nn.RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

        if enable_cond_proj:
            self.cond_proj = nn.Sequential(
                nn.Linear(d_model, d_model)
            )

    def get_norm_and_qkv(self, hidden_states, use_rope, is_rope_2d=False, rope2d_h=16, rope2d_w=16, rope1d_offset=0):
        x_norm = self.norm_in(hidden_states)

        q, k, v = self.qkv_proj(x_norm).chunk(3, dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_head, d=self.d_head)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.n_head, d=self.d_head)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.n_head, d=self.d_head)
        
        q, k = self.norm_q(q), self.norm_k(k)
        
        if use_rope:
            if is_rope_2d:
                q, k = self.rope2d(q, k, rope2d_h, rope2d_w)
            else:
                q, k = self.rope1d(q, k, offset=rope1d_offset)
                
        return x_norm, q, k, v
    
    def forward(self, h=None, x=None, h_mask=None, x_mask=None, x_cond=None, 
                is_causal=False, use_residual=True, 
                use_rope=True, x_is_rope2d=False, h_is_rope2d=False, rope2d_h=16, rope2d_w=16,
                cache=None, return_cache=False):
        if x_cond is not None:
            x = x + torch.sigmoid(self.cond_proj(x)) * x_cond

        if cache is not None:
            x_kv_cache, h_kv_cache = cache.get("x", None), cache.get("hist", None)
        else:
            x_kv_cache, h_kv_cache = None, None

        rope1d_offset = x_kv_cache[0].shape[2] if x_kv_cache is not None else 0
        x_norm, q_x, k_x, v_x = self.get_norm_and_qkv(x, use_rope, x_is_rope2d, rope2d_h, rope2d_w, rope1d_offset)
        if x_kv_cache is not None:
            k_past, v_past = x_kv_cache
            k_x = torch.cat([k_past, k_x], dim=2)
            v_x = torch.cat([v_past, v_x], dim=2)

        new_x_kv_cache = (k_x, v_x) if return_cache else None

        dropout_p = self.dropout if self.training else 0.0

        L_q = q_x.size(2)
        L_k = k_x.size(2)
        if L_q == 1 and rope1d_offset > 0:
            is_causal = False
        if x_mask is not None:
            x_mask = x_mask.view(x.shape[0], 1, 1, -1)
            if is_causal:
                causal_mask = torch.ones((L_q, L_k), dtype=torch.bool, device=q_x.device).tril(diagonal=rope1d_offset)
                x_mask = x_mask & causal_mask.view(1, 1, L_q, L_k)
                is_causal = False # 合并后必须告诉 SDPA 关闭自带的因果掩码     
        attn_x = F.scaled_dot_product_attention(q_x, k_x, v_x, attn_mask=x_mask, dropout_p=dropout_p, is_causal=is_causal)
        
        # 2. 如果存在历史 h，计算交叉注意力并融合
        new_h_kv_cache = h_kv_cache
        if (h_kv_cache is not None) or (h is not None and h.shape[1] > 0):
            if h_kv_cache is not None:
                k_h, v_h = h_kv_cache
            else:
                _, _, k_h, v_h = self.get_norm_and_qkv(h, use_rope, is_rope_2d=h_is_rope2d, rope2d_h=rope2d_h, rope2d_w=rope2d_w, rope1d_offset=0)
                if return_cache:
                    new_h_kv_cache = (k_h, v_h)
            
            # 安全处理 padding mask
            if h_mask is not None:
                safe_h_mask = h_mask.bool().clone()
                safe_h_mask[:, 0] = True # 防止全 False 时底层 SDPA 抛出 NaN
                sdpa_h_mask = safe_h_mask.view(x.shape[0], 1, 1, -1)
            else:
                sdpa_h_mask = None
                
            attn_x_h = F.scaled_dot_product_attention(q_x, k_h, v_h, attn_mask=sdpa_h_mask, dropout_p=dropout_p, is_causal=False)
            
            # 计算融合系数 Alpha
            alpha_hist = self.alpha_hist_predictor(x_norm)
            alpha_hist = alpha_hist.transpose(1, 2).unsqueeze(-1) # [B, H, L_x, 1]
            
            # 直接使用乘法掐断无历史样本的 alpha，替代繁琐的 if 和 masked_fill
            if h_mask is not None:
                has_hist = h_mask.bool().any(dim=-1).view(-1, 1, 1, 1)
                alpha_hist = alpha_hist * has_hist
            
            # 无历史样本的 alpha_hist 会变成 0，自动保留 100% 的 attn_x，逻辑自洽
            attn_x = alpha_hist * attn_x_h + attn_x
            
        # 3. 输出、门控与残差
        attn_x = rearrange(attn_x, 'b h l d -> b l (h d)')
        attn_x = attn_x * torch.sigmoid(self.gate_attn_proj(x_norm))
        
        x_out = x + self.attn_out_proj(attn_x) if use_residual else self.attn_out_proj(attn_x)

        x_out = x_out + self.ffn(self.norm_ffn(x_out))

        if return_cache:
            new_cache = {"x": new_x_kv_cache, "hist": new_h_kv_cache}
            return x_out, new_cache
        return x_out