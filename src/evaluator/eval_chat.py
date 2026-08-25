import logging
import time
from tqdm import tqdm
import math
import torch

# pylint: disable=import-error
from src.dataset.build import get_data_loader
from src.utils.utils import try_release_gpu_mem, logging_memory
from src.utils.utils import handle_checkpoint, compile_model

from ..utils.setup import get_device, get_writer, get_tokenizer, get_ar_model


def evaluate(config):
    # --- 1. Setup Environment ---
    device = get_device(config)
    writer = get_writer(config)
    
    tokenizer = get_tokenizer(config)
    loader_train, loader_eval = get_data_loader(config)
    
    ar_model = get_ar_model(config, device)

    ar_ckpt_path = f"out/checkpoints/train/{config.experiment_name}/ar.pth"
    handle_checkpoint(ar_model, ckpt_path=ar_ckpt_path, save=False)

    if config.hardware.compile:
        logging.info("Compiling the ar_model with torch.compile()...")
        ar_model = compile_model(ar_model, auto_eager=config.hardware.auto_eager)
        
    curr_model = ar_model
    
    writer.add_text("config/all", f"{config}")

    try:
        curr_model.eval()
        
        total_loss = 0.0
        total_loss_lm = 0.0
        total_loss_aux = 0.0
        total_valid_tokens = 0
        num_batches = 0

        progress_bar = tqdm(loader_eval, desc=f"{config.experiment_name} Eval")
            
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                try_release_gpu_mem(device)

                device_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                input_ids = device_batch.pop('inputs').long()
                targets = device_batch.pop('targets').long()
                
                extra_dict = device_batch
                
                loss_dict = curr_model(input_ids, targets, extra_dict)
                
                loss = loss_dict.get('loss', 0.0)
                loss_lm = loss_dict.get('loss_lm', 0.0)
                loss_aux = loss_dict.get('loss_aux', 0.0)
                
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                loss_lm_val = loss_lm.item() if isinstance(loss_lm, torch.Tensor) else loss_lm
                loss_aux_val = loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux
                
                valid_tokens = (targets != -100).sum().item()
                if valid_tokens == 0:
                    continue

                total_loss += loss_val * valid_tokens
                total_loss_lm += loss_lm_val * valid_tokens
                total_loss_aux += loss_aux_val * valid_tokens
                total_valid_tokens += valid_tokens
                num_batches += 1
                
                current_avg_loss = total_loss / total_valid_tokens
                current_avg_loss_lm = total_loss_lm / total_valid_tokens
                
                try:
                    current_ppl = math.exp(current_avg_loss_lm)
                except OverflowError:
                    current_ppl = float('inf')
                
                bar_info = {
                    'loss': f"{current_avg_loss:.4f}",
                    'lm_loss': f"{current_avg_loss_lm:.4f}",
                    'ppl': f"{current_ppl:.2f}"
                }
                progress_bar.set_postfix(bar_info)

        if total_valid_tokens > 0:
            avg_loss = total_loss / total_valid_tokens
            avg_loss_lm = total_loss_lm / total_valid_tokens
            avg_loss_aux = total_loss_aux / total_valid_tokens
            try:
                ppl = math.exp(avg_loss_lm)
            except OverflowError:
                ppl = float('inf')
        else:
            avg_loss, avg_loss_lm, avg_loss_aux, ppl = 0.0, 0.0, 0.0, float('inf')
        
        logging.info(f"Eval Results - Loss: {avg_loss:.4f}, LM Loss: {avg_loss_lm:.4f}, Aux Loss: {avg_loss_aux:.4f}, PPL: {ppl:.4f}")
        
        # Log to tensorboard (using global step 0 or epoch number if available, here we just log it once)
        writer.add_scalar('Eval/Loss', avg_loss, 0)
        writer.add_scalar('Eval/Loss_LM', avg_loss_lm, 0)
        writer.add_scalar('Eval/Loss_Aux', avg_loss_aux, 0)
        writer.add_scalar('Eval/PPL', ppl, 0)

    except Exception as e:
        # 使用 logging.exception() 记录带有完整回溯的错误
        logging.exception(f"An error occurred during eval: {e}")
    
    logging.info("eval finished.")
    writer.close()
    