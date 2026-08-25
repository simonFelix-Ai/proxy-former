# util/downloader.py
import os
import hashlib
import logging
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def calculate_sha256(file_path, chunk_size=8192):
    """计算文件的 SHA256 哈希值"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_pretrained_weights(weight_config, local_weights_dir):
    if weight_config is None:
        return

    if not weight_config.auto_download:
        return

    if os.path.exists(local_weights_dir):
        if any(f.endswith('.pth') for f in os.listdir(local_weights_dir)):
            return
    
    if weight_config.source == "hf":
        repo_id = weight_config.hf.repo_id
        
        subpath = weight_config.hf.subpath
        is_folder = weight_config.hf.is_folder

        if weight_config.hf.use_mirror:
            mirror_url = weight_config.get("mirror_url")
        else:
            mirror_url = None

        if is_folder:
            logging.info(f"Downloading directory {subpath} from {repo_id}...")
            # 批量下载整个子目录
            local_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{subpath}/*",
                local_dir=local_weights_dir,
                endpoint=mirror_url
            )

            base_dir = Path(local_weights_dir)
            src_path = base_dir / subpath
            for item in src_path.iterdir():
                shutil.move(str(item), str(base_dir / item.name))
            shutil.rmtree(src_path, ignore_errors=True)
        else:
            logging.info(f"Downloading single file {subpath} from {repo_id}...")
            # 下载单个文件
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=subpath,
                local_dir=local_weights_dir,
                endpoint=mirror_url
            )

        logging.info(local_path)
