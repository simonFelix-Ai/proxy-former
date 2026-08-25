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
from src.models.components.proxy_former import ProxyFormer, TransformerBlock

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
        
        # Shared Embedding
        self.x_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.h_embedding = self.x_embedding
        
        self.chunk_size = 1
        
        n_layers = model_args.n_layers
        self.total_layers = n_layers
        
        tokenizer_obj = self.config.tokenizer.handle if self.config.tokenizer else None
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj.tokenizer if hasattr(tokenizer_obj, 'tokenizer') else tokenizer_obj
        else:
            self.tokenizer = None

        self.gen_transformers = nn.ModuleList([
            TransformerBlock(d_model=self.d_model, d_head=64, dropout=self.dropout) 
            for _ in range(n_layers)
        ])

        self.hist_blocks = self.gen_transformers
        
        self.lm_head = nn.Sequential(
            nn.RMSNorm(self.d_model),
            nn.Linear(self.d_model, self.vocab_size, bias=False)
        )
        self.lm_head[-1].weight = self.x_embedding.weight

        self.ig_layer_idx = max(0, (n_layers // 2) - 1)
        self.inter_head = nn.Sequential(
            nn.RMSNorm(self.d_model),
            nn.Linear(self.d_model, self.vocab_size, bias=False)
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

    def get_pred(self, h_seq, x_seq, return_inter=False, ig_scale=1.0):
        pad_id = self.tokenizer.pad_token_id if (self.tokenizer and hasattr(self.tokenizer, 'pad_token_id')) else 0

        if h_seq is None:
            h_mask = None
            h = None         
        else:
            h_mask = (h_seq != pad_id)
            h = self.h_embedding(h_seq)

        x = self.x_embedding(x_seq)

        inter_x = x
        for i in range(self.total_layers):
            x = self.gen_transformers[i](h=h, x=x, h_mask=h_mask, is_causal=True)
            if h_seq is not None:
                h = self.hist_blocks[i](x=h, x_mask=h_mask, is_causal=True)
            
            if i == self.ig_layer_idx:
                inter_x = x

        logits = self.lm_head(x)

        if return_inter or ig_scale > (1.0 + 1e-5):
            logits_inter = self.inter_head(inter_x)
            if return_inter:
                return logits_inter, logits
            else:
                return logits, logits_inter + ig_scale * (logits - logits_inter)
            
        return None, logits
    
    def forward(self, x_seq, targets, extra_dict={}):
        h_seq = extra_dict['history'] if 'history' in extra_dict else None
        
        logits_inter, logits = self.get_pred(h_seq, x_seq, return_inter=True)
        
        logits_inter_ar = logits_inter
        logits_ar = logits

        loss_lm_ar = F.cross_entropy(logits_ar.reshape(-1, logits_ar.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        loss_inter_ar = F.cross_entropy(logits_inter_ar.reshape(-1, logits_inter_ar.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        loss = loss_lm_ar + 0.3 * (loss_inter_ar)
        
        return {
            "loss": loss,
            "loss_lm": loss_lm_ar,
            "loss_aux": loss_inter_ar
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, do_sample=True, history=None, history_mask=None):
        """
        Autoregressive generation loop with streaming.
        Yields newly generated tokens one by one.
        """
        full_input_ids = input_ids

        for _ in range(max_new_tokens):
            logits_inter, logits = self.get_pred(history, full_input_ids, ig_scale=1.0)
            next_token_logits = logits[:, -1, :]
            
            if do_sample:
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            yield next_token
            
            full_input_ids = torch.cat((full_input_ids, next_token), dim=1)