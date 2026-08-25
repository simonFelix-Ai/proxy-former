import logging
import time
from tqdm import tqdm
import torch
import os
from torchvision.utils import save_image
import torch_fidelity
import shutil
import datetime



# pylint: disable=import-error
from src.dataset.build import get_data_loader
from src.utils.utils import try_release_gpu_mem, logging_memory
from src.utils.utils import setup_train_step, handle_checkpoint, compile_model, ModelEma

from ..utils.setup import get_device, get_writer, get_ae_model, get_fm_model

@torch.no_grad()
def evaluate_image(config, model, inputs, targets, step, nfe=50):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    was_training = raw_model.training
    raw_model.eval()  # 关闭 Dropout, 开启 eval

    device = targets.device

    labels = inputs

    # 2. 模型采样
    with torch.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
        x_gen = raw_model.generate(config, labels, sample_method=config.eval.sample_method, nfe_steps=nfe, ig_scale=config.eval.ig_scale)

    real_imgs = ((targets.detach().cpu() + 1.0) / 2.0).clamp(0, 1)
    fake_imgs = ((x_gen.detach().cpu() + 1.0) / 2.0).clamp(0, 1)

    if was_training:
        raw_model.train()

    return real_imgs, fake_imgs

def evaluate(config):
    """
    Main training orchestrator.
    """    
    # --- 1. Setup Environment ---
    device = get_device(config)
    writer = get_writer(config)
    
    loader_train, loader_eval = get_data_loader(config)

    fm_model = get_fm_model(config, device)
    fm_ckpt_path = f"out/checkpoints/train/{config.experiment_name}/fm.pth"
    
    orig_model = fm_model._orig_mod if hasattr(fm_model, "_orig_mod") else fm_model
    fm_model_ema = ModelEma(orig_model, decay=config.fm_model.ema_decay, use_warmup=True)
    handle_checkpoint(fm_model_ema, ckpt_path=f"{fm_ckpt_path}.ema", save=False)

    if config.fm_model.use_ae:
        ae_model = get_ae_model(config, device)
        ae_ckpt_path = f"out/checkpoints/train/{config.experiment_name}/ae.pth"
        handle_checkpoint(ae_model, ckpt_path=ae_ckpt_path, save=False)
        ae_model.eval()
    
    writer.add_text("config/all", f"{config}")

    max_eval_batches = getattr(config.eval, "max_batches", None)

    save_dir = f"out/eval/{config.experiment_name}/metrics"
    real_dir = os.path.join(save_dir, "real")
    fake_dir = os.path.join(save_dir, "fake")
    if os.path.exists(real_dir):
        shutil.rmtree(real_dir)
    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)        
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
  

    img_idx = 0
    
    try:
        progress_bar = tqdm(loader_eval, desc="Generating Images")
        
        with torch.inference_mode():    
            for i, batch in enumerate(progress_bar):
                if max_eval_batches is not None and i >= max_eval_batches:
                    break

                try_release_gpu_mem(device)

                device_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                inputs = device_batch.pop('inputs').long()
                targets = device_batch.pop('targets').float()

                real_imgs, fake_imgs = evaluate_image(config=config, model=fm_model_ema.shadow_model,
                    inputs=inputs, targets=targets, step=0, nfe=config.eval.nfe_steps
                )

                for k in range(real_imgs.size(0)):
                    save_image(real_imgs[k], os.path.join(real_dir, f"{img_idx:06d}.png"))
                    save_image(fake_imgs[k], os.path.join(fake_dir, f"{img_idx:06d}.png"))
                    img_idx += 1
        
    except Exception as e:
        # 使用 logging.exception() 记录带有完整回溯的错误
        logging.exception(f"An error occurred during eval: {e}")


    logging.info("Starting metrics evaluation via torch_fidelity Python API...")

    del fm_model, fm_model_ema
    if config.fm_model.use_ae:
        del ae_model

    try_release_gpu_mem(device)

    metrics = torch_fidelity.calculate_metrics(
        input1=fake_dir,
        input2=real_dir,
        cuda=(device.type == "cuda"),
        fid=True,
        isc=True,
        verbose=True,  # 开启详细日志输出（可以看到进度条和下载提示）
    )

    # 3. 提取结果
    fid_val = metrics.get("frechet_inception_distance")
    isc_val = metrics.get("inception_score_mean")
    
    logging.info(f"Evaluation Results -> FID: {fid_val:.4f}, ISC: {isc_val:.4f}")
    
    # 4. 保存日志
    metrics_log_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_log_path, "a") as f:
        f.write(f"\n{'='*20} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*20}\n")
        f.write(f"eval config: {config.eval.to_dict()} \n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    
    logging.info("Eval finished.")
    writer.close()
    