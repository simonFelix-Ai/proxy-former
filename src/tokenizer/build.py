# src/utils.py
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import importlib
import sys
import hashlib
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import kornia.filters as K
from torch.nn import functional as F
from typing import Optional


def get_tokenizer(config):
    """
    Dynamically imports and creates a model based on the configuration.
    It looks for a factory function named 'create_<model_name>_model'
    in the corresponding module 'src.models.<model_name>'.
    """
    
    try:
        module_path = config.tokenizer.module
        class_name = config.tokenizer.class_name
        # Dynamically import the module
        logging.info(f"Creating tokenizer using factory: {class_name}")
        
        # 1. 动态导入模块
        module = importlib.import_module(module_path)
        
        # 2. 从模块中获取类对象
        target_class = getattr(module, class_name)
        
        # 3. 实例化类
        tokenizer = target_class(config)
        
    except (ImportError, AttributeError) as e:
        logging.error(f"Could not find or use model factory for '{module_path}'. ")
        raise e

    return tokenizer
