import yaml
import logging
import math
import sys
from typing import Tuple, Dict, Any, Optional

# ======================================================================================
# Config Helper
# ======================================================================================
class YamlConfig:
    """
    A helper class to convert a nested dictionary (from YAML) into an object 
    with dot-notation access. 
    Includes auto-casting for numbers (e.g., "1e-3" -> 0.001).
    """
    def __init__(self, source):
        """
        Args:
            source: Can be a file path string (e.g., "config.yaml") 
                    OR a dictionary (used for recursion).
        """
        # 1. 判断输入来源
        if isinstance(source, str):
            config_dict = self._load_from_file(source)
        elif isinstance(source, dict):
            config_dict = source
        else:
            raise TypeError(f"YamlConfig expected str or dict, got {type(source)}")
        
        # 2. 递归转换并尝试类型推断
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, YamlConfig(value))
            else:
                # ★★★ 修复点：尝试将字符串转为数字 ★★★
                parsed_value = self._try_parse_number(value)
                setattr(self, key, parsed_value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, YamlConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @staticmethod
    def _try_parse_number(value):
        """
        Attempts to convert a string to int or float.
        Handles scientific notation (e.g., "1e-3" -> 0.001).
        """
        if not isinstance(value, str):
            return value
        
        # 尝试转 int
        try:
            return int(value)
        except ValueError:
            pass

        # 尝试转 float (支持 1e-3, 0.001 等)
        try:
            return float(value)
        except ValueError:
            pass
            
        # 如果都不是，保持原样（如 "relu", "path/to/file"）
        return value

    @staticmethod
    def _load_from_file(yaml_path: str):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                # yaml.safe_load 通常能自动处理不带引号的 1e-3，
                # 但如果用户在 yaml 里写了引号 "1e-3"，这里读出来就是 str
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config from {yaml_path}: {e}")
            sys.exit(1)
            
if __name__ == "__main__":
    # 1. 模拟一个 YAML 配置文件的内容 (通常你会从文件 load)    
    # 2. 解析 YAML 到 Config 对象
    config = YamlConfig("test.yaml")
    
    print(f"Config loaded. Hidden Size: {config.model.model_args.d_model}")