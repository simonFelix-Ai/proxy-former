import logging
import os
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast

class GptTokenizer:
    def __init__(self, config):
        tokenizer_config = config.tokenizer
        self.config = config
                
        self.tokenizer_dir = Path(config.tokenizer.tokenizer_dir)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)        
        
        self.tokenizer = None
        self.load()

    def load(self):
        if self.tokenizer is None:
            save_directory = self.tokenizer_dir
            tokenizer_json_file = save_directory / "tokenizer.json"

            if tokenizer_json_file.is_file():
                self.tokenizer = GPT2TokenizerFast.from_pretrained(str(save_directory))
                logging.info(f"Tokenizer 加载成功，Max Length: {self.tokenizer.model_max_length}")
            else:
                special_tokens_dict = {
                    'unk_token': '<UNK>',
                    'bos_token': '<BOS>',
                    'eos_token': '<EOS>', # 覆盖GPT2的 '<|endoftext|>'
                    'pad_token': '<EOS>', # GPT2没有pad token，通常将其指向eos token
                    'additional_special_tokens': ['<PATH>', '<FILE_CONTENT>']
                }
                
                self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", model_max_length=1000000000)
                
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
                logging.info(f"Added {num_added_toks} special tokens: {special_tokens_dict}")
                
                self.tokenizer.save_pretrained(str(save_directory))
                logging.info(f"Tokenizer 已成功保存到: {save_directory}")
            
            self.vocab_size = len(self.tokenizer)
            logging.info(f"vocab size: {self.vocab_size}")

    # ================= 关键修改位置 =================
    def encode(self, text: str, add_eos: bool = False, add_special_tokens: bool = True, **kwargs) -> List[int]:        
        if add_eos:
            text += self.tokenizer.eos_token

        # 将 add_special_tokens 和 **kwargs 透传给底层的 HuggingFace tokenizer
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return ids
    # ================================================

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return decoded_text

    # 自动透传属性（如 pad_token_id, eos_token_id 等）到底层 HuggingFace tokenizer
    def __getattr__(self, name):
        if self.tokenizer is not None and hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def get_tokenizer_gpt2(config):
    return GptTokenizer(config)
