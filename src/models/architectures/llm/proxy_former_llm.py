# pylint: disable=too-many-positional-arguments
"""
This file
"""
import logging
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from torch.amp import autocast

# pylint: disable=import-error
from src.models.components.proxy_former import ProxyFormer
from src.models.components.transformer import TransformerBlock

class ArLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_args = config.ar_model
        d_attn_scale = model_args.d_attn_scale
        
        self.d_model = model_args.d_model
        self.d_attn = self.d_model * d_attn_scale
        self.d_head = model_args.d_head
        self.n_head = max(1, self.d_attn // self.d_head)
        self.dropout = model_args.dropout
        
        self.vocab_size = config.tokenizer.vocab_size
        d_history = model_args.d_history

        share_proxy_gen_transformer = model_args.share_proxy_gen_transformer

        self.rope_enable = model_args.rope_enable

        self.chunk_size = model_args.chunk_size
        
        self.layer_compress_ratio = model_args.layer_compress_ratio
        
        n_layers = model_args.n_layers
        self.total_layers = n_layers
        
        d_fine = d_history
        d_proxy = self.d_model        

        tokenizer_obj = self.config.tokenizer.handle if self.config.tokenizer else None
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj.tokenizer if hasattr(tokenizer_obj, 'tokenizer') else tokenizer_obj
        else:
            self.tokenizer = None

        self.pad_id = pad_id = self.tokenizer.pad_token_id if (self.tokenizer and hasattr(self.tokenizer, 'pad_token_id')) else 0
        self.h_embedding = nn.Embedding(self.vocab_size, d_history, padding_idx=pad_id)
        self.x_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=pad_id)

        self.hist_blocks = nn.ModuleList()
        self.gen_transformers = nn.ModuleList()
        for i in range(n_layers):
            gen_block = TransformerBlock(d_model=self.d_model, d_head=self.d_head, dropout=self.dropout)
            self.gen_transformers.append(gen_block)

            proxy_transformer_block = gen_block if share_proxy_gen_transformer else None
            hist_block = ProxyFormer(d_fine, d_proxy, chunk_size=self.layer_compress_ratio[i], d_head_proxy=self.d_head, proxy_transformer_block=proxy_transformer_block)
            self.hist_blocks.append(hist_block)
        
        self.lm_head = nn.Sequential(
            nn.RMSNorm(d_proxy),
            nn.Linear(d_proxy, self.vocab_size, bias=False)
        )
        self.lm_head[-1].weight = self.x_embedding.weight

        self.ig_layer_idx = max(0, (n_layers // 2) - 1)
        self.inter_head = nn.Sequential(
            nn.RMSNorm(d_proxy),
            nn.Linear(d_proxy, self.vocab_size, bias=False)
        )
        self.inter_head[-1].weight = self.x_embedding.weight

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = 0.02
        num_residual_layers = self.config.ar_model.n_layers * getattr(self.config.ar_model, 'thinking_steps', 1)
        scaled_std = std / math.sqrt(max(1, 2 * num_residual_layers))

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
            if hasattr(module, '_is_scaled_std_output') and module._is_scaled_std_output:
                torch.nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
            if hasattr(module, '_is_sigmoid_proj') and module._is_sigmoid_proj:
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 2)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_pred(self, h_seq, x_seq, return_inter=False, ig_scale=1.0, cache=None, return_cache=False):
        pad_id = self.pad_id
        B = x_seq.shape[0]
        use_cache = False

        hist_meta = cache["hist_meta"] if cache is not None else None
        layer_caches = cache["layers"] if cache is not None else None

        if h_seq is not None:
            hist_len = h_seq.shape[1]
            if hist_len == 0:
                h_seq = torch.full((B, self.chunk_size), pad_id, dtype=torch.long, device=h_seq.device)
                hist_len = h_seq.shape[1]

            if (hist_meta is not None) and (hist_meta["seq_len"] == hist_len) and (layer_caches is not None):
                use_cache = True
            else:
                h_proxy = 0
                h_fine_mask = (h_seq != pad_id)
                h = self.h_embedding(h_seq)
                h_fine = h * h_fine_mask.unsqueeze(-1).to(h.dtype)
        else:
            hist_len = 0
            use_cache = (hist_meta is not None and hist_meta["seq_len"] == 0 and layer_caches is not None)

        if not use_cache:
            layer_caches = None
            curr_hist_meta = {"seq_len": hist_len, "masks": []} if return_cache else None
        else:
            curr_hist_meta = hist_meta

        x = self.x_embedding(x_seq)
        curr_layer_caches = [] if return_cache else None

        inter_x = x
        for i in range(self.total_layers):
            if use_cache:
                h_proxy = None
                h_proxy_mask = hist_meta["masks"][i]
            elif h_seq is not None:
                compress_ratio = self.layer_compress_ratio[i]
                h_proxy_mask = h_fine_mask.view(B, hist_len // compress_ratio, compress_ratio).any(dim=-1)
                
                if (i > 0) and (self.layer_compress_ratio[i] != self.layer_compress_ratio[i-1]):
                    h_proxy = 0

                h_fine, h_proxy = self.hist_blocks[i](h_fine, h_proxy, proxy_mask=h_proxy_mask, is_causal=True, use_rope=self.rope_enable)
            else:
                h_proxy = None
                h_proxy_mask = None

            if return_cache and not use_cache:
                curr_hist_meta["masks"].append(h_proxy_mask)

            layer_cache = layer_caches[i] if layer_caches is not None else None
            
            out = self.gen_transformers[i](            
                h=h_proxy, x=x, h_mask=h_proxy_mask, is_causal=True, use_rope=self.rope_enable,
                cache=layer_cache, return_cache=return_cache
            )

            if return_cache:
                x, new_layer_cache = out
                curr_layer_caches.append(new_layer_cache)
            else:
                x = out

            if i == self.ig_layer_idx:
                inter_x = x

        logits = self.lm_head(x)

        logits_inter = None
        final_logits = logits
        if return_inter:
            logits_inter = self.inter_head(inter_x)
            if ig_scale > (1.0 + 1e-5):
                final_logits = logits_inter + ig_scale * (logits - logits_inter)

        curr_cache = {"hist_meta": curr_hist_meta, "layers": curr_layer_caches} if return_cache else None
        return logits_inter, final_logits, curr_cache
    
    def forward(self, x_seq, targets, extra_dict={}):
        h_seq = extra_dict['history'] if 'history' in extra_dict else None
        
        logits_inter_ar, logits_ar, _ = self.get_pred(h_seq, x_seq, return_inter=True)

        if self.training:
            unreduced_loss = F.cross_entropy(logits_ar.reshape(-1, logits_ar.size(-1)), targets.reshape(-1), ignore_index=-100, reduction='none').view_as(targets)
            
            weights = torch.ones_like(unreduced_loss)
            
            k_tokens = getattr(self.config.ar_model, 'prefix_weight_tokens', 8)
            weight_scale = getattr(self.config.ar_model, 'prefix_weight_scale', 3.0)
            
            weights[:, -self.chunk_size : -self.chunk_size + k_tokens] = weight_scale
            
            valid_mask = (targets != -100).float()
            effective_weights = weights * valid_mask
            
            loss_lm_ar = (unreduced_loss * effective_weights).sum() / effective_weights.sum().clamp(min=1e-5)
        else:
            loss_lm_ar = F.cross_entropy(logits_ar.reshape(-1, logits_ar.size(-1)), targets.reshape(-1), ignore_index=-100)

        loss_inter_ar = F.cross_entropy(logits_inter_ar.reshape(-1, logits_inter_ar.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        loss = loss_lm_ar + 0.3 * (loss_inter_ar)
        
        return {
            "loss": loss,
            "loss_lm": loss_lm_ar,
            "loss_aux": loss_inter_ar
        }

    @staticmethod
    def _sample_token(logits, temperature=1.0, top_k=None, do_sample=True):
        """统一的 Token 采样逻辑 (Greedy / Temperature / Top-K)"""
        if do_sample:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, do_sample=True, history=None):
        curr_x = input_ids
        cache = None

        for step in range(max_new_tokens):
            # Prefill 阶段构建初始 Cache，Decode 阶段持续复用并更新 KV Cache
            _, logits, cache = self.get_pred(history, curr_x, ig_scale=1.0, cache=cache, return_cache=True)

            # 采样最新生成的 token 并作为下一步的单个输入
            curr_x = self._sample_token(logits[:, -1, :], temperature, top_k, do_sample)

            yield curr_x
