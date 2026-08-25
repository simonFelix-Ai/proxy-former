import logging
import os
from pathlib import Path
from typing import Iterator, List
from multiprocessing import Pool

from transformers import AutoTokenizer


from tokenizers import Tokenizer



class MinimindTokenizer:
    def __init__(self, config):
        tokenizer_config = config.tokenizer
        self.config = config
                
        self.tokenizer_dir = Path(config.tokenizer.tokenizer_dir)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)        
        
        self.tokenizer = None
        
        self.load()
        self._train_tokenizer()
        logging.info(f"Initialized processor with config: {self.__dict__}")
        
    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            if self.config.training.train_stage == "ae":
                self.vocab_size = self.config.ae_model.vocab_size
            elif self.config.training.train_stage == "fm":
                self.vocab_size = self.config.fm_model.vocab_size    
            else:
                self.vocab_size = self.config.ar_model.vocab_size
            
    def encode(self, text: str, add_eos: bool = False) -> List[int]:        
        if add_eos:
            text += self.tokenizer.eos_token

        ids = self.tokenizer.encode(text)
        
        return ids

    def decode(self, token_ids: List[int]) -> str:
        decoded_text = self.tokenizer.decode(token_ids)
        
        return decoded_text
    
    def _train_tokenizer(self) -> Tokenizer:
        return


    def create_structured_wiki(self, examples):
        ret = [f"{text}<EOS>" for text in  examples['text']]
        
        return ret

    def create_structured_codeparrot(self, examples):
        content_column='content'
        return [
            f"<BOS><PATH>{path}<FILE_CONTENT>{content}<EOS>"
            for path, content in zip(examples['path'], examples[content_column])
        ]
    
    def create_structured_input(self, examples, ):
        if "wikitext" in self.config.data.dataset_path:
            return self.create_structured_wiki(examples)
        
        if "codeparrot" in self.config.data.dataset_path:
            return self.create_structured_codeparrot(examples)
        
        raise ValueError(f"不支持的数据集名称: {self.config.data.dataset_name}，请检查配置。")

    def tokenize_dataset(self, dataset_dir, tokenized_bin_path, split="train"):
        pass

def get_tokenizer_minimind(config):
    return MinimindTokenizer(config)
