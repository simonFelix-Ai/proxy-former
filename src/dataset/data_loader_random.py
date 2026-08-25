import logging
import os
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    """
    随机预训练数据集。完全脱离外部数据集，纯内存生成 [0, vocab_size) 的随机自回归序列。
    
    数据布局:
    ┌─────────────────────────┬──────────────────────┬──────────────┐
    │   history (压缩区)       │ retained chunk (原始) │ targets chunk │
    │   长度 = hist_len        │       input_ids (chunk_size 整数倍)  │
    │                         │ labels: [-100 ...]    [真实下文 tokens]│
    └─────────────────────────┴──────────────────────┴──────────────┘
    """
    def __init__(self, 
                 config,
                 tokenizer, 
                 chunk_size: int, 
                 split: str = "train"):
        super().__init__()
        self.config = config
        self.chunk_size = chunk_size
        self.split = split
        
        # 从配置中获取词表与序列参数
        self.vocab_size = config.tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id if (tokenizer and hasattr(tokenizer, 'pad_token_id')) else 0

        if self.config.data.history_enable:
            self.history_len = config.data.history_len
        else:
            self.history_len = 0
        
        self.retain_chunks = config.data.retain_chunks
        self.retain_chunks_drop_prob = config.data.retain_chunks_drop_prob
        
        self.num_samples = 10000

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # 1. 动态决定当前样本保留的 retained chunks 数量
        n_retain = self.retain_chunks
        
        # 2. 计算各部分长度
        input_chunks = n_retain + 1
        input_len = input_chunks * self.chunk_size
        
        # 总序列长度 = history + input + 1 (最后 +1 用于自回归 targets 错位)
        total_len = self.history_len + input_len + 1
        
        # 3. 生成随机 Token 序列 (避开 pad_token_id)
        # 如果 pad_token_id 在 [0, vocab_size) 内，尽量生成非 pad token
        if 0 <= self.pad_token_id < self.vocab_size and self.vocab_size > 1:
            tokens = torch.randint(1, self.vocab_size, (total_len,), dtype=torch.long)
            # 如果 pad_id 不是 0，简单偏移避免撞车
            tokens[tokens == self.pad_token_id] = (self.pad_token_id + 1) % self.vocab_size
        else:
            tokens = torch.randint(0, self.vocab_size, (total_len,), dtype=torch.long)

        # 4. 切分各区域
        # (1) History 区域
        if self.history_len > 0:
            raw_history = tokens[:self.history_len].clone()
        else:
            raw_history = torch.empty(0, dtype=torch.long)
            
        # (2) input_ids 区域
        input_start = self.history_len
        input_end = input_start + input_len
        x = tokens[input_start:input_end].clone()

        # (4) labels 区域: 仅最后一个 chunk 计算 loss，前置区域置为 -100
        y = torch.full_like(x, -100)
        # target label 自回归错后 1 位
        target_label_start = input_start + n_retain * self.chunk_size + 1
        target_label_end = target_label_start + self.chunk_size
        y[-self.chunk_size:] = tokens[target_label_start:target_label_end].clone()
        
        # 兜底：若 targets 中有 pad_token_id 也排除计算
        y[y == self.pad_token_id] = -100

        if self.config.data.history_enable:
            return {
                "history": raw_history,
                "inputs": x.long(),
                "targets": y.long()
            }
        else:
            return {
                "inputs": x.long(),
                "targets": y.long()
            }
        
# ==========================================
# DataPipeline (纯内存模拟，无磁盘IO)
# ==========================================
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.tokenizer = config.tokenizer.handle

    def get_dataloaders(self) -> tuple:
        chunk_size = self.config.ar_model.chunk_size

        train_dataset = RandomDataset(
            config=self.config,
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            split="train"
        )
        val_dataset = RandomDataset(
            config=self.config,
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            split="validation"
        )
        
        logging.info(f"  [随机数据集] 训练样本数: {len(train_dataset):,}")
        logging.info(f"  [随机数据集] 验证样本数: {len(val_dataset):,}")
        
        num_workers = getattr(self.config.data, 'num_workers', 0)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.per_device_train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.eval.per_device_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_dataloader, val_dataloader


def data_loader_random(config: dict) -> tuple:
    pipeline = DataPipeline(config)
    return pipeline.get_dataloaders()