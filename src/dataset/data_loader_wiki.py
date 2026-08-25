import os
import logging
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


class WikiPretrainDataset(Dataset):
    """
    Wiki 预训练数据集。每行文本作为独立文档。

    数据布局 (假设 retain_chunks=1):
    ┌─────────────────────────┬──────────────────────┬──────────────┐
    │   history (压缩区)       │ retained chunk (原始) │ targets chunk │
    │   可送入压缩模块          │       input_ids (chunk_size 整数倍)  │
    │   hist_len = 整数倍      │ labels: [-100 ...]    [真值 tokens]  │
    └─────────────────────────┴──────────────────────┴──────────────┘
    """
    def __init__(self, 
                 config,
                 tokenizer, 
                 raw_ds,
                 chunk_size: int, 
                 split: str = "train"):
        super().__init__()
        self.tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        self.chunk_size = chunk_size
        self.split = split
        if self.split == "train":
            self.retain_chunks_drop_prob = config.data.retain_chunks_drop_prob
        
        # retain_chunks: input_ids 中保留几个历史 chunk 的原始 token
        #   0 → input_ids = 1 个 chunk (仅 targets)
        #   1 → input_ids = 2 个 chunk (1 retained + 1 targets)
        #   2 → input_ids = 3 个 chunk (2 retained + 1 targets)
        self.retain_chunks = config.data.retain_chunks

        # ==========================================
        # 2. Tokenize (利用 HF Arrow 缓存)
        # ==========================================
        def process_text(examples):
            batch_tokens = []
            batch_num_chunks = []
            for text in examples['text']:
                tokens = self.tokenizer(str(text), add_special_tokens=False).input_ids
                tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
                
                # pad 到 N * chunk_size + 1 (最后 1 个 token 供最后 chunk 的 label 错位)
                pad_len = (chunk_size - (len(tokens) - 1) % chunk_size) % chunk_size
                tokens.extend([self.tokenizer.pad_token_id] * pad_len)

                batch_tokens.append(tokens)
                batch_num_chunks.append((len(tokens) - 1) // chunk_size)
                
            return {"input_ids": batch_tokens, "num_chunks": batch_num_chunks}

        self.ds = raw_ds.map(
            process_text, batched=True, 
            remove_columns=raw_ds.column_names, 
            desc=f"Wiki Tokenizing [{split}]"
        )
        self.ds.set_format(type='torch', columns=['input_ids', 'num_chunks'])

        # ==========================================
        # 3. 构建样本索引 (doc_idx, local_chunk_idx)
        # ==========================================
        output_dir = config.data.output_dir
        os.makedirs(output_dir, exist_ok=True)
        cache_path = f"{output_dir}/{config.data.dataset_name}.chunk{chunk_size}.retain{self.retain_chunks}.{split}.dat"
        
        if os.path.exists(cache_path):
            self.samples = torch.load(cache_path)
        else:
            samples_list = []
            for doc_idx, num_chunks in enumerate(tqdm(self.ds['num_chunks'], desc=f"Building Index [{split}]")):
                num_chunks = int(num_chunks.item())
                for local_idx in range(num_chunks):
                    samples_list.append((doc_idx, local_idx))
            
            self.samples = torch.tensor(samples_list, dtype=torch.int32)
            torch.save(self.samples, cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        doc_idx, local_idx = self.samples[idx].tolist()
        tokens = self.ds[doc_idx]['input_ids']
        
        # 计算 retain 边界
        n_retain = min(local_idx, self.retain_chunks)
        start_chunk = local_idx - n_retain   # history 覆盖 [0, start_chunk)
        
        # ---- history: 送入压缩模块的部分 ----
        hist_end = start_chunk * self.chunk_size
        if start_chunk == 0:
            raw_history = torch.empty(0, dtype=torch.long)
        else:
            raw_history = tokens[:hist_end].clone()
        
        # ---- input_ids: (n_retain + 1) 个 chunk ----
        input_start = hist_end
        input_end = (local_idx + 1) * self.chunk_size
        x = tokens[input_start : input_end].clone()

        if self.split == "train" and n_retain > 0 and self.retain_chunks_drop_prob > 0:
            if random.random() < self.retain_chunks_drop_prob:
                x[:-self.chunk_size] = self.tokenizer.pad_token_id
        
        # ---- labels: 仅最后 1 个 chunk 有真值, 前面为 -100 ----
        y = torch.full_like(x, -100)
        target_label_start = local_idx * self.chunk_size + 1
        target_label_end = (local_idx + 1) * self.chunk_size + 1
        y[-self.chunk_size:] = tokens[target_label_start : target_label_end].clone()
        y[y == self.tokenizer.pad_token_id] = -100
        
        return {
            "history": raw_history,
            "hist_len": raw_history.size(0),
            "input_ids": x.long(),
            "labels": y.long()
        }


class CollateFnWithBucketing:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    def __call__(self, batch):
        chunk_size = self.config.ar_model.chunk_size
        alignment_size = chunk_size * 16
        
        # 1. 找到当前 batch 里最长的 history length
        max_l_in_batch = max(item["hist_len"] for item in batch)
        
        # 2. 向上取整到 alignment_size 的整数倍 (分桶)
        target_hist_len = ((max_l_in_batch + alignment_size - 1) // alignment_size) * alignment_size

        # 3. input_ids 长度 (batch 内可能因 retain_chunks 不同文档位置而不同)
        max_i_len = max(item["input_ids"].size(0) for item in batch)

        batch_histories = []
        batch_input_ids = []
        batch_labels = []

        for item in batch:
            raw_h = item["history"]
            L = item["hist_len"]

            # -- history padding --
            padded_h = torch.full((target_hist_len,), self.tokenizer.pad_token_id, dtype=raw_h.dtype)
            
            if L > 0:
                padded_h[:L] = raw_h
            
            batch_histories.append(padded_h)
            
            # -- input_ids padding (对齐到 max_i_len) --
            raw_i = item["input_ids"]
            raw_l = item["labels"]
            i_len = raw_i.size(0)

            padded_i = torch.full((max_i_len,), self.tokenizer.pad_token_id, dtype=raw_i.dtype)
            padded_l = torch.full((max_i_len,), -100, dtype=raw_l.dtype)
            padded_i[:i_len] = raw_i
            padded_l[:i_len] = raw_l

            batch_input_ids.append(padded_i)
            batch_labels.append(padded_l)

        if self.config.data.history_enable:
            return {
                "history": torch.stack(batch_histories),
                "inputs": torch.stack(batch_input_ids),
                "targets": torch.stack(batch_labels),
            }
        else:
            return {
                "inputs": torch.stack(batch_input_ids),
                "targets": torch.stack(batch_labels),
            }


# ==========================================
# DataPipeline
# ==========================================
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.tokenizer = config.tokenizer.handle

        self.data_cfg = self.config.data
        self.training_cfg = self.config.training
        self.eval_cfg = self.config.eval
        self.model_cfg = self.config.ar_model

        self.dataset_dir = Path(self.data_cfg.dataset_dir)
        self.dataset_path = self.data_cfg.dataset_path
        self.dataset_name = self.data_cfg.dataset_name

    def get_dataset(self, split):
        import gc
        import shutil
        from datasets import load_dataset, load_from_disk

        disk_fname = str(self.dataset_dir / f"ds_{split}")
        cache_dir = str(self.dataset_dir / "cache")

        try:
            raw_ds = load_from_disk(disk_fname)
        except Exception:
            raw_ds = load_dataset(str(self.dataset_path), str(self.dataset_name), split=split, cache_dir=cache_dir)
            raw_ds.save_to_disk(disk_fname)

            del raw_ds
            gc.collect()

            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)

            raw_ds = load_from_disk(disk_fname)

        return raw_ds

    def get_dataloaders(self) -> tuple:
        chunk_size = self.config.ar_model.chunk_size
        
        train_raw_ds = self.get_dataset(split="train")
        val_raw_ds = self.get_dataset(split="validation")

        common_args = {
            "config": self.config,
            "tokenizer": self.tokenizer,
            "chunk_size": chunk_size,
        }        
        
        train_dataset = WikiPretrainDataset(**common_args, split="train", raw_ds=train_raw_ds)
        val_dataset = WikiPretrainDataset(**common_args, split="validation", raw_ds=val_raw_ds)
        
        logging.info(f"  训练集样本数: {len(train_dataset):,}")
        logging.info(f"  验证集样本数: {len(val_dataset):,}")

        collate_fn = CollateFnWithBucketing(
            config=self.config,
            tokenizer=self.tokenizer
        )
        
        num_workers = self.config.data.num_workers
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.per_device_train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.eval.per_device_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False
        )
        
        return train_dataloader, val_dataloader


def data_loader_wiki(config: dict) -> tuple:
    pipeline = DataPipeline(config)
    return pipeline.get_dataloaders()