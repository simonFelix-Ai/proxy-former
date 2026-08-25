import argparse
import logging
import os
import importlib

from src.utils.utils import setup_logging_logger
from src.utils.config import YamlConfig
from src.utils.utils import set_global_seed

def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer ae_model.")
    parser.add_argument(
        '--config', type=str, default='configs/fidelity_ultra_long_context.yaml',
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    return args

def get_eval(config):
    """根据 config 中的字符串动态加载评估函数"""
    module_path = config.eval.module

    # 1. 动态导入模块 (相当于 import src.trainer.train_test as mod)
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        logging.error(f"无法找到模块: {module_path}，请检查 YAML 配置或 sys.path")
        raise e

    # 2. 从模块中获取函数 (如果 YAML 中没指定 func，默认找 "train_tests" 或 "train")
    func_name = config.eval.func

    if not hasattr(mod, func_name):
        raise AttributeError(
            f"模块 {module_path} 中没有找到名为 '{func_name}' 的函数！"
        )

    eval_fn = getattr(mod, func_name)
    return eval_fn

if __name__ == "__main__":
    args = get_args()

    config = YamlConfig(args.config)

    log_dir = os.path.join("out", "log", "eval")
    setup_logging_logger(log_dir, config.experiment_name)
    
    logging.info("config: %s", config.to_dict())
    
    set_global_seed(666)

    eval_fn = get_eval(config)

    eval_fn(config)
    