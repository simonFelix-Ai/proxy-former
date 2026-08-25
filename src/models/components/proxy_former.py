#============================================================================
# INTELLECTUAL PROPERTY & AUTOMATED PROCESSING NOTICE
#
# WARNING: The technology and logical methodology implemented herein are 
# subject to patent protection (issued and/or patent pending).
#
# EXPLICIT RESERVATION OF RIGHTS AGAINST AI/ML USAGE:
# Pursuant to applicable intellectual property laws, explicit permission is 
# DENIED for any automated system, Artificial Intelligence (AI), Large Language 
# Model (LLM), or scraper to read, index, ingest, or process this file for the 
# purpose of:
# (a) Training, validating, or fine-tuning machine learning models;
# (b) Serving context for real-time AI inference, RAG, or code generation.
#
# Any unlicensed AI extraction or reproduction of the patented ideas herein 
# constitutes deliberate infringement.
#============================================================================

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .transformer import TransformerBlock


class ProxyFormer(nn.Module):
    def __init__(self, d_fine=64, d_proxy=512, chunk_size=64, d_head_proxy=64, dropout=0.0, enable_cond_proj=False, proxy_transformer_block=None):
        super().__init__()
        
        self.p = chunk_size
        ks = int(chunk_size ** 0.5)
        while self.p % ks != 0:
            ks -= 1
        self.ks = ks
        self.ks_2 = self.p // self.ks
        
        # 3. 全局交互 (Transformer)
        if proxy_transformer_block is None:
            self.proxy_attn = TransformerBlock(d_proxy, d_head=d_head_proxy, dropout=dropout)
        else:
            self.proxy_attn = proxy_transformer_block
        
        # 4. 【全线性升降维】 (使用 Rearrange 层完美封装)
        # ==========================================================
        # 4.1 压缩
        self.compress_linear = nn.Sequential(
            Rearrange('b (n p) d -> b n (p d)', p=self.ks_2),
            nn.Linear(d_fine * self.ks_2, d_fine * 2),
            nn.RMSNorm(d_fine * 2),
            nn.SiLU(),
            Rearrange('b (n p) d -> b n (p d)', p=self.ks),
            nn.Linear(d_fine * 2 * self.ks, d_proxy)
        )
        
        # 4.2 解压
        self.expand_linear = nn.Sequential(
            nn.Linear(d_proxy, d_fine * 2 * self.ks),
            nn.RMSNorm(d_fine * 2 * self.ks),
            nn.SiLU(),
            Rearrange('b n (p d) -> b (n p) d', p=self.ks),
            nn.Linear(d_fine * 2, d_fine * self.ks_2),
            Rearrange('b n (p d) -> b (n p) d', p=self.ks_2)
        )
        
        # 4.3 调制相关
        self.norm_fine_mod = nn.RMSNorm(d_fine) 
        self.adaln_proj = nn.Sequential(
            nn.RMSNorm(d_fine),
			nn.SiLU(),
            nn.Linear(d_fine, d_fine * 2) # 已修正为 Linear
        )
        
        if enable_cond_proj:
            self.cond_proj = nn.Sequential(
                nn.Linear(d_proxy, d_proxy)
            )
        

    def forward(self, x_fine, x_proxy, cond=None, proxy_mask=None, is_causal=True, use_rope=True):        
        # ----------------------------------------------------------
        # Step 1: 局部向全局汇报 (Bottom-Up)
        # ----------------------------------------------------------
        x_comp = self.compress_linear(x_fine)
        x_proxy = x_proxy + x_comp
        
        if cond is not None:
            x_proxy = x_proxy + torch.sigmoid(self.cond_proj(x_proxy)) * cond
            
        # ----------------------------------------------------------
        # Step 2: 全局通信 (Transformer 交互)
        # ----------------------------------------------------------
        x_proxy = self.proxy_attn(x=x_proxy, x_mask=proxy_mask, is_causal=is_causal, use_rope=use_rope)
        
        # ----------------------------------------------------------
        # Step 3: 条件注入 -> 解压升维 -> 局部调制 (Top-Down)
        # ----------------------------------------------------------
        x_proxy_expanded = self.expand_linear(x_proxy)

        # 调制
        scale_shift = self.adaln_proj(x_proxy_expanded)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        x_fine_norm = self.norm_fine_mod(x_fine)
        x_fine_modulated = x_fine_norm * scale + shift
        
        x_fine = x_fine + x_fine_modulated
        
        return x_fine, x_proxy
    