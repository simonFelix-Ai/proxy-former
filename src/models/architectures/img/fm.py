import logging
import math
import torch
from torch import nn
import torch.nn.functional as F

from src.models.components.proxy_former_img import ProxyFormerImg, ChannelNorm2d
from src.models.components.position_encoding import ScalarEmbedding

class FlowMatchingModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_cfg = config.fm_model
        d_attn_scale = model_cfg.d_attn_scale

        self.d_model = model_cfg.d_model
        self.d_attn = self.d_model * d_attn_scale
        self.d_head = model_cfg.d_head
        self.n_head = max(1, self.d_attn // self.d_head)
        self.dropout = model_cfg.dropout

        d_proxy = self.d_model
        d_fine = model_cfg.d_fine
        n_layers = model_cfg.n_layers
        patch_size = model_cfg.patch_size
        d_head_proxy = model_cfg.d_head

        self.time_sample = model_cfg.time_sample

        self.cfg_enable = model_cfg.cfg_enable
        if self.cfg_enable:
            raise NotImplementedError("cfg is not implemented yet.")

        if config.fm_model.use_ae:
            in_channel = config.ae_model.d_latent
        else:
            in_channel = config.data.img_channels

        self.time_mlp_proxy = nn.Sequential(
            ScalarEmbedding(d_proxy),
            nn.Linear(d_proxy, d_proxy * 2),
            nn.SiLU(),
            nn.Linear(d_proxy * 2, d_proxy),
        )

        self.time_mlp_fine = nn.Sequential(
            ScalarEmbedding(d_fine),
            nn.Linear(d_fine, d_fine * 2),
            nn.SiLU(),
            nn.Linear(d_fine * 2, d_fine),
        )

        self.input_proj = nn.Conv2d(in_channel, d_fine, kernel_size=1)

        self.label_emb = nn.Embedding(config.data.num_classes, self.d_model)

        self.blocks = nn.ModuleList([
            ProxyFormerImg(d_fine, d_proxy, patch_size=patch_size, d_head_proxy=d_head_proxy, enable_cond_proj=True)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.GroupNorm(32, d_fine),
            nn.SiLU(),
            nn.Conv2d(d_fine, in_channel*4, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel*4, in_channel, 3, 1, 1),
        )

        self.ig_layer_idx = max(0, (n_layers // 2) - 1)
        self.inter_head = nn.Sequential(
            nn.GroupNorm(32, d_fine),
            nn.SiLU(),
            nn.Conv2d(d_fine, in_channel*4, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(in_channel*4, in_channel, 3, 1, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        num_residual_layers = self.config.fm_model.n_layers
        scaled_std = std / math.sqrt(max(1, 2 * num_residual_layers))

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
            if (
                hasattr(module, "_is_scaled_std_output")
                and module._is_scaled_std_output
            ):
                torch.nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
            if (
                hasattr(module, "_is_sigmoid_proj")
                and module._is_sigmoid_proj
            ):
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 2)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # =========================================================================
    def forward(self, xt, t, txt_emb, return_inter=False):
        t_emb_proxy = self.time_mlp_proxy(t.float()).unsqueeze(1)
        t_emb_fine = self.time_mlp_fine(t.float()).unsqueeze(2).unsqueeze(2)
        x_fine = self.input_proj(xt)
        x_proxy = 0

        inter_x_fine = None
        for i, block in enumerate(self.blocks):
            x_fine, x_proxy = block(
                x_fine, x_proxy, fine_cond=t_emb_fine, proxy_cond=t_emb_proxy, txt_emb=txt_emb.unsqueeze(1), is_causal=False
            )
            if i == self.ig_layer_idx:
                inter_x_fine = x_fine

        x_pred = self.head(x_fine)

        if return_inter:
            x_pred_inter = self.inter_head(inter_x_fine)
            return x_pred, x_pred_inter
        
        return x_pred

    # =========================================================================
    # 2. 推理引导封装 (Inference-time Guidance)
    # =========================================================================
    def predict_x(self, xt, t, txt_emb, ig_scale=1.0):
        if ig_scale <= 1.0 + 1e-5:
            return self.forward(xt, t, txt_emb, return_inter=False)

        x_pred, x_pred_inter = self.forward(xt, t, txt_emb, return_inter=True)
        # 利用中间层和最终层的差值进行信息引导 (IG) 外推
        return x_pred_inter + ig_scale * (x_pred - x_pred_inter)

    def sample_time(self, batch, device) -> torch.Tensor:
        """Sample integration time t with the selected scheme."""
        args = self.time_sample
        if args.method == "uniform":
            t = torch.rand(batch, device=device)
        elif args.method == "stratified":
            # One sample per equal-probability stratum: lower variance than iid uniform.
            t = (torch.arange(batch, device=device).float() + torch.rand(batch, device=device)) / max(batch, 1)
        elif args.method == "beta":
            d = torch.distributions.Beta(args.time_beta_a, args.time_beta_b)
            t = d.sample((batch,)).to(device)
        elif args.method == "logit_normal":
            t = torch.randn(batch, device=device) * args.time_ln_std + args.time_ln_mean
            t = torch.sigmoid(t)
        elif args.method == "mid_heavy":
            # Emphasize the nonlinear middle of the trajectory.
            u = torch.rand(batch, device=device)
            t = 0.5 + args.time_mid_bias * (u - 0.5)
        else:
            raise ValueError(args.method)
        return t.clamp(0.0, 1.0)

    # =========================================================================
    # 4. 训练损失接口 (Training / Loss API)
    # =========================================================================
    def compute_loss(self, labels, targets, extra_dict={}):
        B = targets.shape[0]
        device = targets.device

        if self.config.fm_model.use_ae:
            ae_model = self.config.ae_model.handle
            with torch.no_grad():
                z_latent_2d = ae_model.encode(targets)
                x1 = z_latent_2d
        else:
                x1 = targets

        x0 = torch.randn_like(x1)

        t = self.sample_time(B, device)
        t_exp = t.view(-1, 1, 1, 1)
        xt = (1.0 - t_exp) * x0 + t_exp * x1  # Flow Matching 线性插值

        x_target = x1

        label_emb = self.label_emb(labels)

        x_pred, x_pred_inter = self.forward(xt, t, label_emb, return_inter=True)

        # 直接计算 MSE Loss
        loss_final = nn.functional.mse_loss(x_pred, x_target)
        loss_inter = nn.functional.mse_loss(x_pred_inter, x_target)
        loss = loss_final + 0.5 * loss_inter
        loss = 1.0 * loss

        return {
            "loss": loss,
            "loss_final": loss_final,
            "loss_aux": loss_inter
        }
    
    @torch.no_grad()
    def sample_heun(self, config, labels, nfe_steps=10, ig_scale=1.0):
        """使用 Heun 积分法 (Heun's Method / 2阶预测-校正法) 从噪声生成图像"""
        B = labels.size(0)
        device = next(self.parameters()).device

        if config.fm_model.use_ae:
            shape = (config.ae_model.d_latent, 
                    config.ae_model.latent_h, 
                    config.ae_model.latent_w)
        else:
            shape = (config.data.img_channels, 
                    config.data.img_h, 
                    config.data.img_w)            

        txt_emb = self.label_emb(labels)
        x_gen = torch.randn(B, *shape, device=device)

        for i in range(nfe_steps):
            # 1. 当前时间 t_i
            t_val = torch.full((B,), i / nfe_steps, device=device)

            # --- 步骤 A: 预测步 (Predictor / Euler step) ---
            x_pred_1 = self.predict_x(
                x_gen, t_val, txt_emb, ig_scale=ig_scale
            )
            # 计算当前时刻的速度步长 d1 = v1 * dt
            d1 = (x_pred_1 - x_gen) / (nfe_steps - i)
            x_next_pred = x_gen + d1

            # 如果是最后一步，直接使用预测值，避免对 t=1.0 处做除零评估
            if i == nfe_steps - 1:
                x_gen = x_next_pred
            else:
                # --- 步骤 B: 校正步 (Corrector step) ---
                # 下一个时间 t_{i+1}
                t_next_val = torch.full((B,), (i + 1) / nfe_steps, device=device)
                
                x_pred_2 = self.predict_x(
                    x_next_pred, t_next_val, txt_emb, ig_scale=ig_scale
                )
                # 计算预测时刻的速度步长 d2 = v2 * dt
                d2 = (x_pred_2 - x_next_pred) / (nfe_steps - (i + 1))
                
                # 取两次速度步长的平均值更新 x_gen (Heun 2阶校正)
                x_gen = x_gen + 0.5 * (d1 + d2)

        if config.fm_model.use_ae:
            ae_model = config.ae_model.handle
            img_recon = ae_model.decode(x_gen)
        else:
            img_recon = x_gen

        return img_recon

    @torch.no_grad()
    def sample_euler(self, config, labels, nfe_steps=10, ig_scale=1.0):
        """使用欧拉积分法 (Euler Method) 从噪声生成图像"""
        B = labels.size(0)
        device = next(self.parameters()).device

        if config.fm_model.use_ae:
            shape = (config.ae_model.d_latent, 
                    config.ae_model.latent_h, 
                    config.ae_model.latent_w)
        else:
            shape = (config.data.img_channels, 
                    config.data.img_h, 
                    config.data.img_w)            

        txt_emb = self.label_emb(labels)
        x_gen = torch.randn(B, *shape, device=device)

        for i in range(nfe_steps):
            t_val = torch.full((B,), i / nfe_steps, device=device)

            x_pred = self.predict_x(
                x_gen, t_val, txt_emb, ig_scale=ig_scale
            )

            x_gen = x_gen + (x_pred - x_gen) / (nfe_steps - i)

        if config.fm_model.use_ae:
            ae_model = config.ae_model.handle
            img_recon = ae_model.decode(x_pred)
        else:
            img_recon = x_gen

        return img_recon


    @torch.no_grad()
    def generate(self, config, labels, sample_method, nfe_steps=10, ig_scale=1.0):
        if sample_method == "euler":
            return self.sample_euler(config, labels, nfe_steps, ig_scale)   
        elif sample_method == "heun":
            return self.sample_heun(config, labels, nfe_steps, ig_scale)