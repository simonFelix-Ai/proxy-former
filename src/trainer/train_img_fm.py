import logging
import time
from tqdm import tqdm
import torch
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# pylint: disable=import-error
from src.dataset.build import get_data_loader
from src.utils.utils import try_release_gpu_mem, logging_memory
from src.utils.utils import setup_train_step, handle_checkpoint, compile_model, ModelEma

from ..utils.setup import get_device, get_writer, get_ae_model, get_fm_model, get_optimizer, get_grad_accum, get_profiler_ctx, get_lr_scheduler


class TrainLogger:
    def __init__(self, writer, logging_steps: int, info_log_interval: int = 1000):
        self.writer = writer
        self.logging_steps = logging_steps
        self.info_log_interval = info_log_interval

    def log(self, step: int, loss_dict: dict, lr_scheduler, progress_bar):
        if step % self.logging_steps != 0:
            return

        current_loss = loss_dict['loss'].item()
        current_lr = lr_scheduler.get_last_lr()[0]

        # TensorBoard 记录
        self.writer.add_scalar('Loss/train', current_loss, step)
        self.writer.add_scalar('LearningRate', current_lr, step)

        # 构建展示信息
        bar_info = {
            'step': step,
            'lr': f"{current_lr:.3e}",
            'final': f"{loss_dict['loss_final']:.4f}",
            'aux': f"{loss_dict['loss_aux']:.4f}",
            'total': f"{current_loss:.4f}"
        }

        # 刷新 UI & 日志
        progress_bar.set_postfix(bar_info)
        if step % self.info_log_interval == 0:
            logging.info(bar_info)


@torch.no_grad()
def evaluate_image(config, model, inputs, targets, step, nfe=50):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    was_training = raw_model.training
    raw_model.eval()  # 关闭 Dropout, 开启 eval

    device = targets.device
    num_classes = config.data.num_classes
    num_samples = 10

    # 1. 从 0 ~ num_classes-1 中随机抽取 10 个类别（优先无放回抽样）
    if num_classes >= num_samples:
        labels = torch.randperm(num_classes, device=device)[:num_samples]
    else:
        labels = torch.randint(0, num_classes, (num_samples,), device=device)

    # 2. 模型采样
    with torch.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
        x_gen = raw_model.sample_euler(config, labels, nfe_steps=nfe, ig_scale=config.eval.ig_scale)

    # 3. 归一化到 [0, 1] 并转为 (N, H, W, C) 用于可视化
    display_imgs = ((x_gen.detach().cpu() + 1.0) / 2.0).clamp(0, 1)
    imgs_np = display_imgs.permute(0, 2, 3, 1).float().numpy()
    labels_list = labels.cpu().tolist()

    save_dir = f"out/eval/{config.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fm.step_{step}.nfe{nfe}.png")

    # 4. 使用 Matplotlib 绘制 2 行 5 列的网格，并在每个图上方标注 Label
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
    for idx, ax in enumerate(axes.flat):
        if idx < len(imgs_np):
            img = imgs_np[idx]
            if img.shape[-1] == 1:  # 灰度图
                ax.imshow(img.squeeze(-1), cmap="gray")
            else:                   # RGB 彩色图
                ax.imshow(img)
            
            # 在图片正上方标注类别 ID
            ax.set_title(f"Label: {labels_list[idx]}", fontsize=11, fontweight="bold")
        ax.axis("off")  # 隐藏坐标轴

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)  # 及时关闭画布，释放内存

    if was_training:
        raw_model.train()

def train(config):
    """
    Main training orchestrator.
    """    
    # --- 1. Setup Environment ---
    device = get_device(config)
    writer = get_writer(config)

    train_logger = TrainLogger(writer, logging_steps=config.training.logging_steps)
    
    loader_train, loader_eval = get_data_loader(config)
    
    grad_accum_steps = get_grad_accum(config, loader_train)

    num_train_epochs = config.training.num_train_epochs

    fm_model = get_fm_model(config, device)
    optimizer, base_optimizer = get_optimizer(fm_model, config)
    lr_scheduler = get_lr_scheduler(config, loader_train, base_optimizer, grad_accum_steps)
    save_every_steps = max(1, int(num_train_epochs * len(loader_train) * config.training.save_tokens_ratio))
    logging.info(f"Checkpoint saving interval: {save_every_steps} steps (ratio: {config.training.save_tokens_ratio})")

    fm_ckpt_path = f"out/checkpoints/train/{config.experiment_name}/fm.pth"
    train_step = setup_train_step(
        model=fm_model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        grad_accum_steps=grad_accum_steps, # 注意：这里的 steps 用于除法缩放
        max_grad_norm=config.training.grad_clip,
        autocast_enabled=config.hardware.autocast,
        ckpt_path=fm_ckpt_path,
        save_every_steps=save_every_steps
    )
    
    if config.hardware.compile:
        logging.info("Compiling the fm_model with torch.compile()...")
        fm_model = compile_model(fm_model, auto_eager=config.hardware.auto_eager)
        
    curr_model = fm_model

    orig_model = curr_model._orig_mod if hasattr(curr_model, "_orig_mod") else curr_model
    fm_model_ema = ModelEma(orig_model, decay=config.fm_model.ema_decay, use_warmup=True)
    handle_checkpoint(fm_model_ema, ckpt_path=f"{fm_ckpt_path}.ema")

    if config.fm_model.use_ae:
        ae_model = get_ae_model(config, device)
        ae_ckpt_path = f"out/checkpoints/train/{config.experiment_name}/ae.pth"
        handle_checkpoint(ae_model, ckpt_path=ae_ckpt_path, save=False)
        ae_model.eval()
    
    writer.add_text("config/all", f"{config}")
    
    # --- 5. Main Training Loop ---
    progress_epoch_bar = tqdm(range(1, num_train_epochs + 1), desc="Overall Training Progress")

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

                    inputs = device_batch.pop('inputs').long()
                    targets = device_batch.pop('targets').float()
                    
                    extra_dict = device_batch

                    def cal_loss():
                        nonlocal loss_dict
                        loss_dict = curr_model.compute_loss(inputs, targets, extra_dict)
                        return loss_dict['loss']
                    
                    _, updated = train_step(cal_loss)

                    if updated:
                        model_profile.step()
                        fm_model_ema.update()

                    current_step = train_step.get_step_counter()
                    if current_step % config.eval.eval_image_steps == 0:
                        logging.info(f"Generating evaluation image at step {current_step}...")
                        evaluate_image(config=config, model=fm_model_ema.shadow_model,
                            inputs=inputs, targets=targets, step=current_step, nfe=config.eval.nfe_steps
                        )
                        evaluate_image(config=config, model=fm_model_ema.shadow_model,
                            inputs=inputs, targets=targets, step=current_step, nfe=1
                        )
                        evaluate_image(config=config, model=fm_model_ema.shadow_model,
                            inputs=inputs, targets=targets, step=current_step, nfe=10
                        )                        
                    train_logger.log(
                        step=current_step,
                        loss_dict=loss_dict,
                        lr_scheduler=lr_scheduler,
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
    