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
from einops import rearrange

class ChannelNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim) 
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')

class ProxyFormerImg(nn.Module):
    def __init__(self, d_fine=64, d_proxy=512, patch_size=16, d_head_proxy=64, dropout=0.0, enable_cond_proj=False):
        super().__init__()
        
        self.p = patch_size
        ks = int(patch_size ** 0.5)
        while self.p % ks != 0:
            ks -= 1
        self.ks = ks
        self.ks_2 = self.p // self.ks
        
        # 3. 全局交互 (Transformer)
        self.proxy_attn = TransformerBlock(d_proxy, d_head=d_head_proxy, dropout=dropout)
        
        # 4. 【全卷积升降维】：Compress -> Expand -> Modulate
        # ==========================================================
        # 4.1 压缩：用于提取局部信息，上报给 Global (Bottom-up 使用)
        self.compress_conv = nn.Sequential(
            nn.Conv2d(d_fine, d_fine * 2, kernel_size=self.ks_2, stride=self.ks_2),
            ChannelNorm2d(d_fine * 2),
            nn.SiLU(),
            nn.Conv2d(d_fine * 2, d_proxy, kernel_size=self.ks, stride=self.ks)
        )
        
        # 4.2 解压：【修改重点】现在用于将 Global 特征先放大回 256x256
        self.expand_conv = nn.Sequential(
            nn.ConvTranspose2d(d_proxy, d_fine * 2, kernel_size=self.ks, stride=self.ks),
            ChannelNorm2d(d_fine * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(d_fine * 2, d_fine, kernel_size=self.ks_2, stride=self.ks_2)
        )
        
        self.norm_fine_mod = ChannelNorm2d(d_fine) # 给高维画布准备的 Norm
        # 注意！因为在升维后生成参数，输入维度变成了 d_fine，需要用 Conv2d
        self.adaln_proj = nn.Sequential(
            ChannelNorm2d(d_fine),
            nn.SiLU(),
            nn.Conv2d(d_fine, d_fine * 2, kernel_size=1)
        )
        
        if enable_cond_proj:
            self.fine_cond_proj = nn.Sequential(
                ChannelNorm2d(d_fine),
                nn.SiLU(),
                nn.Conv2d(d_fine, d_fine, kernel_size=1)
            )
            self.proxy_cond_proj = nn.Sequential(
                nn.RMSNorm(d_proxy),
                nn.Linear(d_proxy, d_proxy)
            )
        
    def forward(self, x_fine, x_proxy, fine_cond=None, proxy_cond=None, txt_emb=None, txt_mask=None, is_causal=False):
        B, C, H, W = x_fine.shape
        grid_h, grid_w = H // self.p, W // self.p
        
        # ----------------------------------------------------------
        # Step 1: 局部向全局汇报 (Bottom-Up)
        # ----------------------------------------------------------
        # 先利用 compress_conv 提取局部浓缩特征 [B, d_proxy, 16, 16]
        x_comp_2d = self.compress_conv(x_fine)
        
        fine_summary = rearrange(x_comp_2d, 'b d h w -> b (h w) d')
        x_proxy = x_proxy + fine_summary

        if proxy_cond is not None:
            x_proxy = x_proxy + self.proxy_cond_proj(proxy_cond)
        
        # ----------------------------------------------------------
        # Step 2: 全局通信
        # ----------------------------------------------------------
        x_proxy = self.proxy_attn(x=x_proxy, h=txt_emb, h_mask=txt_mask, is_causal=is_causal, 
                                  use_rope=True, x_is_rope2d=True, h_is_rope2d=False, rope2d_h=grid_h, rope2d_w=grid_w)

        # ----------------------------------------------------------
        # Step 3: 先升维，再调制 (Top-Down: Expand -> Modulate)
        # ----------------------------------------------------------
        # 3.1 将全局 Token 重新摆放为 2D 形状
        x_proxy_2d_normed = rearrange(x_proxy, 'b (h w) d -> b d h w', h=grid_h, w=grid_w)
        x_proxy_expanded = self.expand_conv(x_proxy_2d_normed)
        if fine_cond is not None:
            x_proxy_expanded = x_proxy_expanded + fine_cond
        scale_shift = self.adaln_proj(x_proxy_expanded) # [B, d_fine*2, 256, 256]
        scale, shift = scale_shift.chunk(2, dim=1)       # 各自[B, d_fine, 256, 256]
        
        # 3.4 像素级调制原始的高清画布 (x_fine)
        x_fine_norm = self.norm_fine_mod(x_fine)

        x_fine_modulated = x_fine_norm * scale + shift

        # 当t接近1时，给 x_fine_modulated 一个快速接近0的机会。
        if fine_cond is not None:
            x_fine_modulated = x_fine_modulated * self.fine_cond_proj(fine_cond)

        # 3.5 残差连接更新画布
        x_fine = x_fine + x_fine_modulated
        
        return x_fine, x_proxy
