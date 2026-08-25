import logging
import os
import torch
import platform

# pylint: disable=import-error
from src.dataset.build import get_data_loader
from src.utils.utils import create_model, get_tensorboard_logger, setup_logging_logger
from src.utils.utils import setup_train_step, handle_checkpoint, compile_model, dump_model, create_optimizer, set_global_seed
from src.utils.weights import get_pretrained_weights 

def get_device(config):
    """Returns the torch device and logs its detailed model name."""
    device_str = config.hardware.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 获取具体设备型号
    if device.type == "cuda":
        # torch.cuda.get_device_name() 支持传入 device 对象或整型索引 (如 0, 1)
        device_model = torch.cuda.get_device_name(device)
        logging.info("Using device: %s (%s)", device, device_model)
    elif device.type == "mps":
        logging.info("Using device: %s (Apple Silicon GPU)", device)
    else:
        # 获取 CPU 架构/型号信息
        cpu_model = platform.processor() or platform.machine()
        logging.info("Using device: %s (%s)", device, cpu_model)

    return device

def get_writer(config):
    log_dir = os.path.join("out", "tensorboard")
    writer = get_tensorboard_logger(log_dir, config.experiment_name)
    return writer

def get_tokenizer(config):
    from src.tokenizer.build import get_tokenizer as get_tokenizer_low
    tokenizer = get_tokenizer_low(config)
    config.tokenizer.handle = tokenizer
    config.tokenizer.vocab_size = tokenizer.vocab_size
    logging.info("Vocabulary size: %d", config.tokenizer.vocab_size)
    return tokenizer

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """将字符串转换为对应的 torch.dtype"""
    DTYPE_MAP = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    dtype_str = str(dtype_str).lower()
    return DTYPE_MAP.get(dtype_str, torch.bfloat16)  # 找不到时默认使用 bfloat16

def get_ae_model(config, device):
    """Creates and returns the ae_model on the specified device."""
    ae_model = create_model(config, config.ae_model).to(device)
    logging.info("Model: %s", config.ae_model.class_name)
    dump_model(ae_model)
    config.ae_model.handle = ae_model
    checkpoint_dir = f"out/checkpoints/train/{config.experiment_name}"
    get_pretrained_weights(config.ae_model.get("weights"), checkpoint_dir)
    return ae_model

def get_ar_model(config, device):
    """Creates and returns the ar_model on the specified device."""
    ar_model = create_model(config, config.ar_model).to(device)
    logging.info("Model: %s", config.ar_model.class_name)
    dump_model(ar_model)
    config.ar_model.handle = ar_model
    checkpoint_dir = f"out/checkpoints/train/{config.experiment_name}"
    get_pretrained_weights(config.ar_model.get("weights"), checkpoint_dir)
    return ar_model
	
def get_fm_model(config, device):
    """Creates and returns the fm_model on the specified device."""
    fm_model = create_model(config, config.fm_model).to(device)
    logging.info("Model: %s", config.fm_model.class_name)
    dump_model(fm_model)
    config.fm_model.handle = fm_model
    checkpoint_dir = f"out/checkpoints/train/{config.experiment_name}"
    get_pretrained_weights(config.fm_model.get("weights"), checkpoint_dir)
    return fm_model

def get_optimizer(ae_model, config):
    """Creates and returns the optimizer and base_optimizer."""
    logging.info(f"Using optimizer: {config.training.optimizer}.")
    # Create a single optimizer for all parameters
    optimizer, base_optimizer = create_optimizer(ae_model, config)
    return optimizer, base_optimizer

def get_grad_accum(config, loader_train):
    grad_accum_steps = config.training.gradient_accumulation
    logging.info(f"grad_accum_steps: {grad_accum_steps}")
    return grad_accum_steps

def get_profiler_ctx(config):
    if config.profiler.enable:
        # wait: 等待1步
        # warmup: 预热1步
        # active: 真正分析的3步
        # repeat: 重复2次这个周期
        wait = config.get("profiler").get("wait")
        warmup = config.get("profiler").get("warmup")
        active = config.get("profiler").get("active")
        repeat = config.get("profiler").get("repeat")
        profile_schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        
        log_dir = config.get("profiler").get("log_dir") + "/" + config.get("experiment_name")
    
        # 开启时：使用真正的 torch.profiler
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=profile_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    else:
        class NoOpProfiler:
            """一个什么都不做的 Profiler 占位符"""
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            
            def step(self):
                pass
        
        # 关闭时：使用我们定义的哑对象
        profiler_ctx = NoOpProfiler()
    
    return profiler_ctx

def get_lr_scheduler(config, loader, base_optimizer, grad_accum_steps):
    if not config.training.lr_scheduler.enable:
        return None
    
    num_optimizer_steps = len(loader) * config.training.num_train_epochs // grad_accum_steps
    num_warmup_steps = round(num_optimizer_steps * config.training.lr_scheduler.warmup_ratio)
    logging.info(f"Total optimizer steps: {num_optimizer_steps}, Warmup steps: {num_warmup_steps}")

    if config.training.lr_scheduler.type == "warmup_then_cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        min_lr = config.training.lr_scheduler.min_lr

        warmup_scheduler = LinearLR(base_optimizer, start_factor=1e-3, end_factor=1.0, total_iters=num_warmup_steps)
        cosine_scheduler = CosineAnnealingLR(base_optimizer, T_max=num_optimizer_steps - num_warmup_steps, eta_min=min_lr)

        lr_scheduler = SequentialLR(
            base_optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
        
        return lr_scheduler
