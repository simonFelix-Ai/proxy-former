import os
import logging
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Dataset 定义
# ==========================================
class PasskeyRetrievalDataset(Dataset):
    """
    多 Passkey 大海捞针 Dataset。
    从预编码的 token bin 文件 (np.memmap) 中随机采样背景 token，
    均匀插入 N 个 passkey，AR 区域拼接所有 QA 对。
    """
    def __init__(self, tokenizer, token_memmap, is_train, config):
        self.tokenizer = tokenizer
        self.tokens = token_memmap          # np.memmap, dtype=uint16/int32
        self.total_tokens = len(token_memmap)
        self.is_train = is_train

        # 灵活获取配置参数，优先读 data 节点，再读根节点
        data_cfg = getattr(config, 'data', config)
        training_cfg = getattr(config, 'training', config)
        
        self.num_passkeys = data_cfg.num_passkeys
        self.history_len = data_cfg.history_len
        self.batch_size = training_cfg.per_device_train_batch_size
        self.config = config
        self.needle_cache = {}  # 缓存已编码的 Passkey token，避免重复 encode

    def __len__(self):
        # 虚拟数据集长度（适配按 Epoch/Step 训练）
        return 10000 * self.batch_size

    def __getitem__(self, idx):
        if self.is_train and (random.random() < 0.2):
            sample_len = int(10.0 * random.uniform(0.0, 1.0))
        else:
            sample_len = self.history_len
            
        if self.total_tokens >= sample_len:
            start = random.randint(0, self.total_tokens - sample_len)
            bg_tokens = self.tokens[start : start + sample_len].astype(np.int64).tolist()
        else:
            repeats = (sample_len // self.total_tokens) + 2
            repeated_tokens = np.tile(self.tokens, repeats)
            start = random.randint(0, len(repeated_tokens) - sample_len)
            bg_tokens = repeated_tokens[start : start + sample_len].astype(np.int64).tolist()

        # 2. 生成 N 个 passkey 并打乱编号顺序
        n = self.num_passkeys
        segment_len = max(1, len(bg_tokens) // n)

        pk_indices = list(range(1, n + 1))
        random.shuffle(pk_indices)

        passkey_map = {} # 保存 {编号 id: 随机密码 pk}
        insert_positions = []

        for i in range(n):
            pk_id = pk_indices[i] # 取得打乱后的编号
            pk_val = str(random.randint(10000, 99999))
            passkey_map[pk_id] = pk_val

            seg_start = i * segment_len
            seg_end = (i + 1) * segment_len if i < n - 1 else len(bg_tokens)
            insert_pos = random.randint(seg_start, max(seg_start, seg_end - 1))
            
            insert_positions.append((insert_pos, pk_id, pk_val))

        # 优化：升序排序，顺序拼接
        insert_positions.sort(key=lambda x: x[0])
        final_tokens = []
        last_pos = 0
        for insert_pos, pk_id, pk_val in insert_positions:
            final_tokens.extend(bg_tokens[last_pos:insert_pos])
            
            needle_str = f"\nThe passkey_{pk_id} is {pk_val}. Remember it.\n"
            if needle_str not in self.needle_cache:
                self.needle_cache[needle_str] = self.tokenizer.encode(needle_str, add_special_tokens=False)
            
            final_tokens.extend(self.needle_cache[needle_str])
            last_pos = insert_pos
            
        final_tokens.extend(bg_tokens[last_pos:])
        bg_tokens = final_tokens

        history = torch.tensor(bg_tokens, dtype=torch.long)

        # 3. 构造 AR 区域：随机抽取 1 个 Passkey 编号进行提问
        target_id = random.choice(pk_indices)
        target_pk = passkey_map[target_id]

        q_text = f"\nWhat is the passkey_{target_id}? The passkey_{target_id} is"
        a_text = f" {target_pk}."

        q_ids = self.tokenizer.encode(q_text, add_special_tokens=False)
        q_len = len(q_ids)

        full_ar_ids = self.tokenizer.encode(q_text + a_text, add_special_tokens=False)

        input_ids = torch.tensor(full_ar_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:q_len] = -100

        # 4. 错位处理 (Shift)
        input_ids = input_ids[:-1]
        labels = labels[1:]

        return {
            "history": history,
            "input_ids": input_ids,
            "labels": labels
        }


# ==========================================
# 2. CollateFn 定义
# ==========================================
class CollateFnWithBucketing:
    def __init__(self, pad_token_id, chunk_size=4):
        self.pad_token_id = pad_token_id
        self.alignment_size = chunk_size

    def __call__(self, batch):
        max_h_len = max(len(item["history"]) for item in batch)
        # history 长度对齐到 alignment_size 的整数倍
        target_hist_len = max(self.alignment_size, ((max_h_len + self.alignment_size - 1) // self.alignment_size) * self.alignment_size)
        max_i_len = max(len(item["input_ids"]) for item in batch)
        
        batch_histories, batch_h_masks = [], []
        batch_input_ids, batch_i_masks, batch_labels = [], [], []

        for item in batch:
            raw_h = item["history"]
            padded_h = torch.full((target_hist_len,), self.pad_token_id, dtype=raw_h.dtype)
            h_mask = torch.zeros(target_hist_len, dtype=torch.bool)
            if len(raw_h) > 0:
                padded_h[:len(raw_h)] = raw_h
                h_mask[:len(raw_h)] = True
            batch_histories.append(padded_h)
            batch_h_masks.append(h_mask)
            
            raw_i = item["input_ids"]
            raw_l = item["labels"]
            padded_i = torch.full((max_i_len,), self.pad_token_id, dtype=raw_i.dtype)
            i_mask = torch.zeros(max_i_len, dtype=torch.bool)
            padded_l = torch.full((max_i_len,), -100, dtype=raw_l.dtype)
            
            if len(raw_i) > 0:
                padded_i[:len(raw_i)] = raw_i
                i_mask[:len(raw_i)] = True
                padded_l[:len(raw_l)] = raw_l
                
            batch_input_ids.append(padded_i)
            batch_i_masks.append(i_mask)
            batch_labels.append(padded_l)
            
        return {
            "history": torch.stack(batch_histories),
            "history_mask": torch.stack(batch_h_masks),
            "inputs": torch.stack(batch_input_ids),
            "input_mask": torch.stack(batch_i_masks),
            "targets": torch.stack(batch_labels)
        }


# ==========================================
# 3. DataPipeline 修改
# ==========================================
class DataPipeline:
    def __init__(self, config):
        self.config = config

        self.data_cfg = self.config.data
        self.training_cfg = self.config.training
        self.eval_cfg = self.config.eval
        self.model_cfg = self.config.ar_model

        self.dataset_dir = Path(self.data_cfg.dataset_dir)
        self.dataset_path = self.data_cfg.dataset_path
        self.dataset_name = self.data_cfg.dataset_name

        self.train_bin = Path(self.data_cfg.train_bin)
        self.eval_bin = Path(self.data_cfg.eval_bin)

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.train_bin.parent, exist_ok=True)
        os.makedirs(self.eval_bin.parent, exist_ok=True)

        self.prepare_tokenizer()
        self.process_and_build_bins()

    def prepare_tokenizer(self):
        self.tokenizer = self.config.tokenizer.handle

    def download(self, split):
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

    def _tokenize_split_to_bin(self, split: str, target_bin_path: Path):
        if target_bin_path.exists():
            logging.info(f"找到已存在的二进制缓存: '{target_bin_path}'，跳过 Tokenize。")
            return

        logging.info(f"开始 Tokenize 处理 ({split}) 并写入到 '{target_bin_path}'...")

        raw_split_name = "train" if split == "train" else "validation"
        raw_ds = self.download(raw_split_name)

        def process_text(examples):
            batch_tokens = []
            for text in examples["text"]:
                if text.strip():
                    batch_tokens.append(self.tokenizer.encode(text))
            return {"input_ids": batch_tokens}

        ds = raw_ds.map(
            process_text,
            batched=True,
            remove_columns=raw_ds.column_names,
            desc=f"Tokenizing {split}"
        )
        ds = ds.filter(lambda x: len(x["input_ids"]) > 0)

        all_ids = []
        for item in tqdm(ds, desc=f"Concatenating {split} tokens"):
            ids = item["input_ids"]
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            all_ids.extend(ids)

        dtype = np.uint16 if self.tokenizer.vocab_size < 2**16 else np.int32
        arr = np.array(all_ids, dtype=dtype)
        arr.tofile(str(target_bin_path))
        logging.info(f"成功保存 {len(arr):,} tokens 到 '{target_bin_path}'")

    def process_and_build_bins(self):
        self._tokenize_split_to_bin(split="train", target_bin_path=self.train_bin)
        self._tokenize_split_to_bin(split="validation", target_bin_path=self.eval_bin)

    def get_dataloaders(self) -> tuple:
        """
        根据构建好的二进制文件生成 PasskeyRetrievalDataset 和应用了 Bucketing Collate 的 DataLoaders。
        """
        logging.info("=" * 50)
        logging.info("构建 Passkey Retrieval DataLoaders...")

        itemsize = 2 if self.tokenizer.vocab_size < 2**16 else 4
        dtype = np.uint16 if self.tokenizer.vocab_size < 2**16 else np.int32

        train_ratio = getattr(self.training_cfg, 'train_ratio', 1.0)
        eval_ratio = getattr(self.eval_cfg, 'eval_ratio', 1.0)

        total_train_tokens = os.path.getsize(self.train_bin) // itemsize
        need_train_tokens = int(total_train_tokens * train_ratio)

        total_val_tokens = os.path.getsize(self.eval_bin) // itemsize
        need_val_tokens = int(total_val_tokens * eval_ratio)

        logging.info(f"  - 训练集 Total Tokens: {total_train_tokens:,} | 使用: {need_train_tokens:,}")
        logging.info(f"  - 验证集 Total Tokens: {total_val_tokens:,} | 使用: {need_val_tokens:,}")

        # ----------------------------------------------------
        # 1. 使用 np.memmap 打开二进制文件得到 token_memmap
        # ----------------------------------------------------
        train_memmap = np.memmap(str(self.train_bin), dtype=dtype, mode='r')[:need_train_tokens]
        eval_memmap = np.memmap(str(self.eval_bin), dtype=dtype, mode='r')[:need_val_tokens]

        # ----------------------------------------------------
        # 2. 构建 PasskeyRetrievalDataset
        # ----------------------------------------------------
        train_dataset = PasskeyRetrievalDataset(
            tokenizer=self.tokenizer,
            token_memmap=train_memmap,
            is_train=True,
            config=self.config
        )

        val_dataset = PasskeyRetrievalDataset(
            tokenizer=self.tokenizer,
            token_memmap=eval_memmap,
            is_train=False,
            config=self.config
        )

        # ----------------------------------------------------
        # 3. 构建 CollateFnWithBucketing
        # ----------------------------------------------------
        # 获取 pad_token_id (如果无 pad_token 则退回 eos_token_id 或 0)
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, 'eos_token_id', 0)

        # 对齐尺寸，优先读取 ar_model.chunk_size，如无则默认 4
        chunk_size = self.model_cfg.chunk_size

        collate_fn = CollateFnWithBucketing(
            pad_token_id=pad_token_id,
            chunk_size=chunk_size
        )

        # ----------------------------------------------------
        # 4. 构建 DataLoader 并传入 collate_fn
        # ----------------------------------------------------
        num_workers = self.data_cfg.num_workers
        batch_size_train = self.training_cfg.per_device_train_batch_size
        batch_size_eval = self.config.eval.per_device_eval_batch_size

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,             # <--- 关键点：传入自定义 collate_fn
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,             # <--- 关键点：传入自定义 collate_fn
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        logging.info("DataLoader (Passkey + Bucketing) 加载成功！")
        logging.info("=" * 50)

        return train_dataloader, val_dataloader


def data_loader_passkey_retrieval(config: dict) -> tuple:
    pipeline = DataPipeline(config)
    return pipeline.get_dataloaders()