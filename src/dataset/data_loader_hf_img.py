import os
import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def get_transformed_dataset(config, raw_ds):
    import torchvision.transforms as transforms
    
    # 1. 定义图像变换
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")), # 强制转 RGB
        transforms.Resize((config.data.img_h, config.data.img_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, config, hf_ds, transform):
            self.config = config
            self.ds = hf_ds
            self.transform = transform

        def __len__(self):
            return len(self.ds)
        
        def __getitem__(self, idx):
            item = self.ds[idx]
            
            return {
                "inputs": item[self.config.data.inputs_key],
                "targets": self.transform(item[self.config.data.targets_key]),
            }

    ds = HFDatasetWrapper(config, raw_ds, transform)
    return ds

# ==========================================
# DataPipeline
# ==========================================
class DataPipeline:
    def __init__(self, config):
        self.config = config

        self.data_cfg = self.config.data
        self.training_cfg = self.config.training
        self.eval_cfg = self.config.eval

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
        train_raw_ds = self.get_dataset(split=self.config.data.train_split)
        val_raw_ds = self.get_dataset(split=self.config.data.test_split)
        
        train_dataset = get_transformed_dataset(config=self.config, raw_ds=train_raw_ds)
        val_dataset = get_transformed_dataset(config=self.config, raw_ds=val_raw_ds)
        
        logging.info(f"  训练集样本数: {len(train_dataset):,}")
        logging.info(f"  验证集样本数: {len(val_dataset):,}")

        num_workers = self.config.data.num_workers
        
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


def data_loader_hf_img(config: dict) -> tuple:
    pipeline = DataPipeline(config)
    return pipeline.get_dataloaders()