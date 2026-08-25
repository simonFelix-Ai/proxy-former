import logging
import time
from tqdm import tqdm

# pylint: disable=import-error
from src.dataset.build import get_data_loader
from src.utils.utils import try_release_gpu_mem, logging_memory
from src.utils.utils import setup_train_step, handle_checkpoint, compile_model

from ..utils.setup import get_device, get_writer, get_tokenizer, get_ar_model, get_optimizer, get_grad_accum, get_profiler_ctx, get_lr_scheduler


class TrainLogger:
    def __init__(self, writer, logging_steps: int, info_log_interval: int = 1000):
        self.writer = writer
        self.logging_steps = logging_steps
        self.info_log_interval = info_log_interval

    def log(self, step: int, loss_dict: dict, lr_scheduler, global_tokens: int, progress_bar):
        if step % self.logging_steps != 0:
            return

        current_loss = loss_dict['loss'].item()
        current_lr = lr_scheduler.get_last_lr()[0]

        # TensorBoard 记录
        self.writer.add_scalar('Loss/train', current_loss, step)
        self.writer.add_scalar('LearningRate', current_lr, step)

        # 构建展示信息
        bar_info = {
            'tok': global_tokens,
            'step': step,
            'lr': f"{current_lr:.3e}",
            'lm': f"{loss_dict.get('loss_lm', 0.0):.2f}",
            'aux': f"{loss_dict.get('loss_aux', 0.0):.2f}",
            'loss': f"{current_loss:.2f}"
        }

        # 刷新 UI & 日志
        progress_bar.set_postfix(bar_info)
        if step % self.info_log_interval == 0:
            logging.info(bar_info)

def train(config):
    """
    Main training orchestrator.
    """    
    # --- 1. Setup Environment ---
    device = get_device(config)
    writer = get_writer(config)

    train_logger = TrainLogger(writer, logging_steps=config.training.logging_steps)
    
    tokenizer = get_tokenizer(config)
    loader_train, loader_eval = get_data_loader(config)
    
    grad_accum_steps = get_grad_accum(config, loader_train)

    num_train_epochs = config.training.num_train_epochs

    ar_model = get_ar_model(config, device)
    optimizer, base_optimizer = get_optimizer(ar_model, config)
    lr_scheduler = get_lr_scheduler(config, loader_train, base_optimizer, grad_accum_steps)
    save_every_steps = max(1, int(num_train_epochs * len(loader_train) * config.training.save_tokens_ratio))
    logging.info(f"Checkpoint saving interval: {save_every_steps} steps (ratio: {config.training.save_tokens_ratio})")

    ar_ckpt_path = f"out/checkpoints/train/{config.experiment_name}/ar.pth"
    train_step = setup_train_step(
        model=ar_model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        grad_accum_steps=grad_accum_steps, # 注意：这里的 steps 用于除法缩放
        max_grad_norm=config.training.grad_clip,
        autocast_enabled=config.hardware.autocast,
        ckpt_path=ar_ckpt_path,
        save_every_steps=save_every_steps
    )

    if config.hardware.compile:
        logging.info("Compiling the ar_model with torch.compile()...")
        ar_model = compile_model(ar_model, auto_eager=config.hardware.auto_eager)
        
    curr_model = ar_model
    
    writer.add_text("config/all", f"{config}")

    # --- 5. Main Training Loop ---
    progress_epoch_bar = tqdm(range(1, num_train_epochs + 1), desc="Overall Training Progress")

    total_trained_tokens = 0
    loss_dict = {}

    try:
        for epoch in progress_epoch_bar:
            train_epoch_start_time = time.time()
            
            curr_model.train()

            progress_bar = tqdm(loader_train, desc=f"Epoch {epoch}/{config.training.num_train_epochs} {config.experiment_name}")
            
            profiler_ctx = get_profiler_ctx(config)
            with profiler_ctx as model_profile:            
                for i, batch in enumerate(progress_bar):
                    try_release_gpu_mem(device)

                    device_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                    input_ids = device_batch.pop('inputs').long()
                    targets = device_batch.pop('targets').long()
                    
                    extra_dict = device_batch

                    def cal_loss():
                        nonlocal loss_dict
                        loss_dict = curr_model(input_ids, targets, extra_dict)
                        return loss_dict['loss']
                    
                    _, updated = train_step(cal_loss)

                    total_trained_tokens += input_ids.shape[1] * input_ids.shape[0]

                    if updated:
                        model_profile.step()

                    train_logger.log(
                        step=train_step.get_step_counter(),
                        loss_dict=loss_dict,
                        lr_scheduler=lr_scheduler,
                        global_tokens=total_trained_tokens,
                        progress_bar=progress_bar
                    )
            
            writer.add_scalar(
                'time/train_per_epoch', time.time() - train_epoch_start_time, epoch
            )
    except Exception as e:
        # 使用 logging.exception() 记录带有完整回溯的错误
        logging.exception(f"An error occurred during training: {e}")
    
    logging.info("Training finished.")
    writer.close()
    