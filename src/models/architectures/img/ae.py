import logging
import math
import torch
from torch import nn
import torch.nn.functional as F

from src.models.components.proxy_former_img import ProxyFormerImg, ChannelNorm2d
from src.models.components.position_encoding import ScalarEmbedding

from src.utils.utils import NoiseInjection

class ImageAe(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        model_cfg = config.ae_model
        
        self.d_head = model_cfg.d_head
        dropout = model_cfg.dropout
        
        d_enc_fine = model_cfg.d_enc_fine
        d_dec_fine = model_cfg.d_dec_fine
        d_model_enc = model_cfg.d_model_enc
        d_model_dec = model_cfg.d_model_dec
        n_layers_enc = model_cfg.n_layers_enc
        n_layers_dec = model_cfg.n_layers_dec
        img_channels = config.data.img_channels
        img_h = config.data.img_h
        img_w = config.data.img_w
        latent_h = model_cfg.latent_h
        latent_w = model_cfg.latent_w
        patch_size_per_proxy = model_cfg.patch_size_per_proxy

        self.is_vae = model_cfg.is_vae

        d_latent = model_cfg.d_latent

        compress_ratio = img_h // latent_h

        self.img_spatial_compress = nn.Conv2d(img_channels, d_enc_fine, kernel_size=compress_ratio, stride=compress_ratio)
        
        self.encode_blocks = nn.ModuleList([
            ProxyFormerImg(d_enc_fine, d_model_enc, patch_size=patch_size_per_proxy//compress_ratio, d_head_proxy=self.d_head, dropout=dropout) 
            for _ in range(n_layers_enc)
        ])

        self.latent_noise_enable = model_cfg.latent_noise_enable
        self.add_noise = NoiseInjection()
        self.proj_to_latent = nn.Conv2d(d_enc_fine, d_latent, kernel_size=1)
        self.latent_norm = ChannelNorm2d(d_latent)

        self.img_spatial_expand = nn.ConvTranspose2d(d_latent, d_dec_fine, kernel_size=compress_ratio, stride=compress_ratio)

        self.decode_blocks = nn.ModuleList([
            ProxyFormerImg(d_dec_fine, d_model_dec, patch_size=patch_size_per_proxy, d_head_proxy=self.d_head, dropout=dropout) 
            for _ in range(n_layers_dec)
        ])

        self.head = nn.Sequential(
            ChannelNorm2d(d_dec_fine), nn.SiLU(),
            nn.Conv2d(d_dec_fine, 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, img_channels, 3, 1, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        pass
        
    def encode(self, img):
        x_latent = self.img_spatial_compress(img)
        x_proxy = 0

        for block in self.encode_blocks:
            x_latent, x_proxy = block(x_latent, x_proxy)

        if self.is_vae:
            raise NotImplementedError("VAE mode is not implemented yet.")
        else:
            z_latent = self.proj_to_latent(x_latent)
            z_latent = self.latent_norm(z_latent)
        
        return z_latent

    def decode(self, z_latent):
        x_fine = self.img_spatial_expand(z_latent)
        x_proxy = 0

        for block in self.decode_blocks:
            x_fine, x_proxy = block(x_fine, x_proxy)
            
        img_recon = self.head(x_fine)

        return img_recon

    def forward(self, img):
        z_latent = self.encode(img)

        if self.training and self.latent_noise_enable:
            z_latent = self.add_noise(z_latent)
            
        img_recon = self.decode(z_latent)
        
        return img_recon

    def compute_loss(self, img, extra_dict={}):
        img_recon = self.forward(img)

        loss = F.l1_loss(img_recon, img)
        
        return {
            "loss": loss,
            "aux": loss.detach() 
        }
    