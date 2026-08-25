# 文件路径：src/dataset/build.py
import logging
import importlib

def get_data_loader(config):
    """
    Dynamically imports and creates data loaders based on configuration.
    """
    train_dataloader, val_dataloader = None, None
    
    try:
        module_path = config.data.module
        class_name = config.data.class_name               
        logging.info(f"Building data loader using: {class_name} from {module_path}")
        
        # 1. 动态导入模块
        module = importlib.import_module(module_path)
        
        # 2. 获取类或函数
        target_class = getattr(module, class_name)
        
        # 3. 实例化并获取 dataloaders
        train_dataloader, val_dataloader = target_class(config)
        
    except (ImportError, AttributeError) as e:
        logging.error(f"Could not find or use data loader builder for '{module_path}'.")
        raise e

    return train_dataloader, val_dataloader