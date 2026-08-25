# src/utils.py
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import importlib
import sys
import hashlib
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import kornia.filters as K
from typing import Optional
import torch.nn as nn
import copy

def get_tensorboard_logger(log_dir, name):
    """
    Initializes a TensorBoard SummaryWriter.
    """
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique subdirectory for this run based on the experiment name
    run_dir = os.path.join(log_dir, name)
    
    logging.info(f"TensorBoard log directory: {run_dir}")
    return SummaryWriter(log_dir=run_dir)

def setup_logging_logger(log_dir, name, level=logging.INFO, force=True):
    """
    配置根 logger，使其同时输出到文件和控制台（sys.stdout）。
    完全自包含在一个函数中，无需在外部调用 logging.basicConfig。
    """
    # 1. 创建日志目录和文件路径
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    
    # 2. 定义统一的格式化器 (Formatter)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # 3. 创建 File Handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 4. 创建 Stream Handler (输出到控制台)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # 5. 使用 basicConfig 统一配置根 Logger
    # 在 Python 3.8+ 中，force=True 会自动清理之前存在的 handlers
    logging.basicConfig(
        level=level,
        handlers=[file_handler, stream_handler],
        force=force
    )
    
    logging.info(f"Root logger setup complete. Logging to console and: {os.path.abspath(log_file)}")

def create_model(config, model_handle, vision=False):
    """
    Dynamically imports and creates a model based on the configuration.
    It looks for a factory function named 'create_<model_name>_model'
    in the corresponding module 'src.models.<model_name>'.
    """
        
    try:
        if vision:
            pass
        else:
            module_path = model_handle.module
            class_name = model_handle.class_name
            # Dynamically import the module
            logging.info(f"Creating model using factory: {class_name}")
            
            # 1. 动态导入模块
            module = importlib.import_module(module_path)
            
            # 2. 从模块中获取类对象
            target_class = getattr(module, class_name)
            
            # 3. 实例化类
            model = target_class(config)
        
    except (ImportError, AttributeError) as e:
        logging.error(f"Could not find or use model factory for '{module_path}'. ")
        raise e

    return model

def get_tensor_bytes_recursively(obj):
    """
    递归地遍历一个对象，计算其中所有 PyTorch 张量的总字节数。
    """
    total_bytes = 0
    
    if isinstance(obj, torch.Tensor):
        # Base Case: 如果对象是张量，计算其大小并返回
        return obj.nelement() * obj.element_size()
    
    elif isinstance(obj, dict):
        # Recursive Step: 如果是字典，遍历其值
        for value in obj.values():
            total_bytes += get_tensor_bytes_recursively(value)
            
    elif isinstance(obj, (list, tuple)):
        # Recursive Step: 如果是列表或元组，遍历其元素
        for item in obj:
            total_bytes += get_tensor_bytes_recursively(item)
            
    # 如果是其他类型 (int, str等)，则不增加字节数，直接返回当前的 total_bytes (即0)
    return total_bytes    

def try_release_gpu_mem(device):
    if not torch.cuda.is_available():
        return
    
    total = torch.cuda.get_device_properties(device).total_memory
    rev = torch.cuda.max_memory_reserved(device)
    if rev > (total * 0.96):
        logging.info("Peak GPU Memory Reserved: %.4f GB", rev/(1024**3))
        logging.info("torch.cuda.empty_cache()")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
def logging_memory(device):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
        peak_memory_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
        )
        peak_memory_reserved_gb = (
            torch.cuda.max_memory_reserved(device) / (1024**3)
        )
        
        logging.info(
            "  Peak GPU Memory Allocated: %.4f GB", peak_memory_allocated_gb
        )
        logging.info(
            "  Peak GPU Memory Reserved:  %.4f GB", peak_memory_reserved_gb
        )

# ======================================================================================
# 采样函数 (Sampling Function)
# ======================================================================================

def sample_from_logits(
    logits: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None, # <-- 新增：需要传入历史 token
    sampling: str = "topk",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0, # <-- 新增：重复惩罚参数
) -> torch.Tensor:
    """
    Samples a token from the logits distribution, with support for repetition penalty.

    Args:
        logits (torch.Tensor): The output logits from the language model. 
                               Shape: (batch_size, 1, vocab_size).
        input_ids (Optional[torch.Tensor]): The sequence of previously generated tokens.
                                            Required for repetition_penalty > 1.0.
                                            Shape: (batch_size, sequence_length).
        sampling (str): The sampling strategy. One of "greedy", "random", "topk", "topp".
        temperature (float): Softmax temperature for sampling.
        top_k (int): The number of highest probability tokens to consider for top-k sampling.
        top_p (float): The cumulative probability threshold for nucleus (top-p) sampling.
        repetition_penalty (float): The penalty for repeating tokens. 1.0 means no penalty.
                                    Values > 1.0 discourage repetition.

    Returns:
        torch.Tensor: The predicted token indices. Shape: (batch_size, 1).
    """
    # Ensure logits are in float32 for stable softmax
    logits = logits.float()

    if temperature <= 0:
        temperature = 1.0
        
    # Logits shape is expected to be (B, 1, V), squeeze it to (B, V)
    if logits.dim() == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)

    # ==========================================================
    # === 新增：应用 Repetition Penalty ===
    # ==========================================================
    if repetition_penalty != 1.0:
        if input_ids is None:
            raise ValueError("input_ids must be provided to use repetition_penalty")
        
        batch_size, vocab_size = logits.shape
        
        # Create a score tensor of shape (batch_size, vocab_size)
        # For each batch item, set the penalty score for tokens that appeared in its history
        score = torch.gather(logits, 1, input_ids)
        
        # Apply the penalty:
        # - If score > 0, divide by repetition_penalty to reduce the probability.
        # - If score < 0, multiply by repetition_penalty to make it even more negative, further reducing probability.
        # This is the standard implementation in Hugging Face Transformers.
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        
        # Scatter the updated scores back to the original logits tensor
        logits.scatter_(1, input_ids, score)
    # ==========================================================
    
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    if sampling == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    # For all sampling methods below, we first compute the probabilities
    probs = F.softmax(logits, dim=-1)

    if sampling == "random":
        return torch.multinomial(probs, num_samples=1)

    elif sampling == "topk":
        # Get top-k probabilities and their indices from the *original* logits
        # This is more efficient than applying top-k to the full probability distribution
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        
        # Sample from the filtered distribution
        topk_probs = F.softmax(values, dim=-1)
        choice = torch.multinomial(topk_probs, num_samples=1)
        
        # Map the choice back to the original vocabulary indices
        return torch.gather(indices, dim=-1, index=choice)

    elif sampling == "topp":
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create a mask for tokens to remove (those exceeding top_p)
        mask_to_remove = cumulative_probs > top_p
        mask_to_remove[..., 1:] = mask_to_remove[..., :-1].clone()
        mask_to_remove[..., 0] = 0

        # Create a new probs tensor and apply the mask
        # We need to use the original indices to apply the mask correctly
        # An easier way is to set the logits of removed tokens to -inf before softmax
        
        # Alternative (and more common) implementation for top-p:
        indices_to_remove = mask_to_remove.scatter(1, sorted_indices, mask_to_remove)
        logits[indices_to_remove] = -float("Inf")
        
        final_probs = F.softmax(logits, dim=-1)
        
        return torch.multinomial(final_probs, num_samples=1)

    else:
        raise ValueError(f"Unknown sampling method: '{sampling}'")
    
def human_readable_format(num: int) -> str:
    """
    将一个数字格式化为人类可读的字符串，支持 K, M, B 单位。
    """
    if num == 0:
        return "0"
    if num < 1_000:
        return str(num)
    
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    
    suffixes = ['', 'K', 'M', 'B', 'T'] # 支持到 T (Trillion)
    
    # 根据 magnitude 选择后缀，并格式化数字
    # %.2f 会将 1.3 格式化为 1.30，而 %g 会移除末尾多余的0，如 1.30 -> 1.3
    formatted_num = f"{num:.2f}".rstrip('0').rstrip('.')
    
    return f"{formatted_num}{suffixes[magnitude]}"

class NoiseInjection(torch.nn.Module):
    def __init__(self, min_noise_scale=0.0, max_noise_scale=0.1, dim=-1):
        """
        自适应噪声注入层。
        根据输入的标准差 (std) 动态调整噪声幅度，确保噪声与信号强度成比例。
        
        Args:
            min_noise_scale (float): 最小噪声比例 (相对于输入std)。
            max_noise_scale (float): 最大噪声比例。
            dim (int): 计算标准差的维度。对于 (B, L, D) 数据，通常为 -1。
        """
        super().__init__()
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.dim = dim

    def forward(self, x):
        # 1. 训练模式检查 (通常只在训练时加噪)
        if not self.training:
            return x
                    
        # 2. 计算噪声 (不需要梯度，纯数值计算)
        with torch.no_grad():
            # 动态构建广播形状: (B, 1, 1, ...) 根据 x 的维度自动调整
            # 例如 x是(B, L, D) -> shape变成 (B, 1, 1)
            # 例如 x是(B, D)    -> shape变成 (B, 1)
            broadcast_shape = [x.shape[0]] + [1] * (x.ndim - 1)
            
            # 生成随机噪声比例
            noise_scale = torch.rand(broadcast_shape, device=x.device)
            noise_scale = noise_scale * (self.max_noise_scale - self.min_noise_scale) + self.min_noise_scale
            
            # 使用 self.dim 而不是硬编码 -1
            std = x.std(dim=self.dim, keepdim=True) # 按 Token (Token-wise)计算标准差，或者全局 x.std() 也可以（对所有batch进行计算）
            
            # 叠加
            noise = torch.randn_like(x) * std * noise_scale
            
        # 3. 返回
        return x + noise

class ModelEma(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999, use_warmup: bool = True):
        super().__init__()
        self.target_decay = decay
        self.use_warmup = use_warmup
        
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        
        # 1. 拷贝模型
        self.shadow_model = copy.deepcopy(model)
        
        # 强制将 EMA 的影子模型转换为 float32，避免精度截断
        self.shadow_model.to(torch.float32)
        
        self.shadow_model.requires_grad_(False)
        self.shadow_model.eval()

        self._model_ref = [model]

    @property
    def model(self):
        return self._model_ref[0]

    @torch.no_grad()
    def update(self):
        self.num_updates += 1
        step = self.num_updates.item()
        
        if self.use_warmup:
            decay = min(self.target_decay, (1.0 + step) / (10.0 + step))
        else:
            decay = self.target_decay

        # 运算时将主模型的参数临时转为 float32 后，再加到 EMA 参数上
        for shadow_param, param in zip(self.shadow_model.parameters(), self.model.parameters()):
            if param.requires_grad:
                shadow_param.data.mul_(decay).add_(param.data.to(torch.float32), alpha=1.0 - decay)

        # 同步 Buffer 时，浮点数转 float32，整数保持原样
        for shadow_buffer, buffer in zip(self.shadow_model.buffers(), self.model.buffers()):
            if buffer.is_floating_point():
                shadow_buffer.copy_(buffer.to(torch.float32))
            else:
                shadow_buffer.copy_(buffer)


def handle_checkpoint(model, optimizer=None, lr_scheduler=None, scaler=None, ckpt_path=None, save=True, step_getter=None):
    import atexit
    import os
    import torch

    if ckpt_path is None:
        script_name = os.path.basename(globals().get('__file__', 'model.py'))
        ckpt_path = f"./checkpoints/final_staged_{script_name}.pth"
        
    meta_ckpt_path = f"{ckpt_path}.meta"

    def save_checkpoint(curr_steps=0):
        if curr_steps == 0 and step_getter is not None:
            curr_steps = step_getter() if callable(step_getter) else 0
            
        fpath = f"{ckpt_path}.{curr_steps}"
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        torch.save(model.state_dict(), fpath)
        print(f"\n[Auto-Save] Model saved to {fpath}")

        # 建立/更新模型的软链接 (指向最新 step 文件)
        if os.path.lexists(ckpt_path):
            os.remove(ckpt_path)
        os.symlink(os.path.basename(fpath), ckpt_path)

        # 保存 Meta 状态 (Optimizer, LR Scheduler, Scaler)
        meta_dict = {}
        meta_dict['steps'] = curr_steps
        if optimizer is not None: meta_dict['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None: meta_dict['lr_scheduler'] = lr_scheduler.state_dict()
        if scaler is not None: meta_dict['scaler'] = scaler.state_dict()
        
        if meta_dict:
            fpath_meta = f"{ckpt_path}.meta.{curr_steps}"
            torch.save(meta_dict, fpath_meta)
            print(f"[Auto-Save] Meta states saved to {fpath_meta}")
            
            # 建立/更新 Meta 状态的软链接
            if os.path.lexists(meta_ckpt_path):
                os.remove(meta_ckpt_path)
            os.symlink(os.path.basename(fpath_meta), meta_ckpt_path)

    # 2. 注册退出钩子（不需要缩进后面的代码）
    if save:
        atexit.register(save_checkpoint)
    
    if not os.path.exists(ckpt_path):
        return save_checkpoint, 0
        
    pretrained_dict = torch.load(ckpt_path)
    model_dict = model.state_dict()
    
    # 移除 _orig_mod. 前缀
    # 同时处理可能存在的 module. 前缀 (防止以后用 DDP)
    clean_dict = {}
    for k, v in pretrained_dict.items():
        new_k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_dict[new_k] = v
    
    # 过滤并匹配
    matched_dict = {}
    for k, v in clean_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                print(f"[Skip] {k}: Shape mismatch! {v.shape} vs {model_dict[k].shape}")
        # else: 可以打印模型中不存在的 Key，通常是一些过时的参数
                
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {ckpt_path}: {len(matched_dict)}/{len(model_dict)} params.")
    
    # 尝试加载对应的 Meta 状态
    last_steps = 0
    if os.path.exists(meta_ckpt_path):
        try:
            meta_dict = torch.load(meta_ckpt_path, map_location='cpu')
            last_steps = meta_dict['steps']
            if optimizer is not None and 'optimizer' in meta_dict:
                optimizer.load_state_dict(meta_dict['optimizer'])
            if lr_scheduler is not None and 'lr_scheduler' in meta_dict:
                lr_scheduler.load_state_dict(meta_dict['lr_scheduler'])
            if scaler is not None and 'scaler' in meta_dict:
                scaler.load_state_dict(meta_dict['scaler'])
            print(f"Successfully loaded meta states from {meta_ckpt_path}")
        except Exception as e:
            print(f"fail: {e}")

    return save_checkpoint, last_steps

def setup_train_step(model, optimizer,
                     lr_scheduler=None, grad_accum_steps=1, max_grad_norm=10.0, 
                     autocast_enabled=True, dtype=torch.bfloat16,
                     ckpt_path=None,
                     save_every_steps=10000):
    from torch.amp import GradScaler, autocast
    
    # 逻辑优化：只要是 float16 且启用了 AMP，就必须用 scaler
    use_scaler = (dtype == torch.float16 and autocast_enabled)
    scaler = GradScaler(enabled=use_scaler)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    _step_counter = 0

    checkpoint_save_fn, _step_counter = handle_checkpoint(
        model, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler, 
        ckpt_path=ckpt_path,
        step_getter=lambda: _step_counter
    )
    
    # 初始化时先清空一次梯度，防止残留
    optimizer.zero_grad()

    def step(loss_compute_fn, optimizer_closure=None):
        nonlocal _step_counter
        _step_counter += 1
        
        # 1. 前向传播
        with autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
            loss = loss_compute_fn()
            # 缩放 Loss 用于累积
            scaled_loss = loss / grad_accum_steps

        # 2. 反向传播
        if use_scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # 3. 更新判断
        updated = False
        if _step_counter % grad_accum_steps == 0:
            # 3.1 梯度裁剪
            
            if max_grad_norm > 1e-5:
                # 重要：在使用 scaler 时，裁剪前必须先 unscale
                if use_scaler:
                    scaler.unscale_(optimizer)
            
                if _step_counter % 2000 == 0:  # 防止刷屏
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                    print(f"total gradient norm: {total_norm:.4f}")
                    
                # 执行裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if optimizer_closure is not None:
                def wrap_optimizer_closure():
                    with autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
                        with torch.no_grad(): # 线搜索时不需要再存计算图
                            return optimizer_closure()                
                if use_scaler:
                    loss = scaler.step(optimizer, wrap_optimizer_closure, loss)
                    scaler.update()
                else:
                    loss = optimizer.step(wrap_optimizer_closure, loss)
            else:
                # 3.2 更新权重
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
            optimizer.zero_grad()
            updated = True

        # 4. 【新增】：如果 _step_counter 到达 save_every_steps 的倍数，触发定时保存
        if _step_counter % save_every_steps == 0:
            checkpoint_save_fn(_step_counter)
            
        # 返回原始 loss 和更新状态
        return loss, updated

    step.get_step_counter = lambda: _step_counter

    return step

def compile_model(model, auto_eager=False):
    if not auto_eager:
        import sys
        torch.set_float32_matmul_precision('high') 
        sys.setrecursionlimit(10000)
        model = torch.compile(model)
        return model
    else:
        import sys
        torch.set_float32_matmul_precision('high') 
        sys.setrecursionlimit(10000)

        # 1. 开启 Dynamo 错误防护（遇到无法编译的图自动降级回 Eager，不崩溃）
        torch._dynamo.config.suppress_errors = True

        # 2. 安全设置 Inductor 优化配置（去除错误路径，防止 AttributeError）
        try:
            # 正确路径是 torch._inductor.config.mix_dim_reduction
            torch._inductor.config.mix_dim_reduction = False
        except AttributeError:
            pass

        try:
            # 关闭导致 SymPy 分数断言崩溃的坐标下降调优
            torch._inductor.config.coordinate_descent_tuning = False
        except AttributeError:
            pass

        # 3. 编译模型（推荐 dynamic=None 让系统自适应，避免全局 dynamic=True 引发 SymPy Bug）
        model = torch.compile(model)
        return model

def human_readable_format(num: int) -> str:
    """
    将一个数字格式化为人类可读的字符串，支持 K, M, B 单位。
    """
    if num == 0:
        return "0"
    if num < 1_000:
        return str(num)
    
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    
    suffixes = ['', 'K', 'M', 'B', 'T'] # 支持到 T (Trillion)
    
    # 根据 magnitude 选择后缀，并格式化数字
    # %.2f 会将 1.3 格式化为 1.30，而 %g 会移除末尾多余的0，如 1.30 -> 1.3
    formatted_num = f"{num:.2f}".rstrip('0').rstrip('.')
    
    return f"{formatted_num}{suffixes[magnitude]}"

def dump_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", human_readable_format(total_params))

def create_mano_exclude(model):
    exclude_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # Mano 只处理 2D 矩阵
        # 且通常排除 Embedding 和 LayerNorm 的参数（即使它们是 2D 或 1D）
        if p.ndim == 2 and "embed" not in name.lower() and "head" not in name.lower():
            pass
        else:
            # 所有的 1D (bias, norm), 3D (conv), 4D 参数都走 AdamW
            exclude_params.append(p)
            
    return exclude_params

def create_optimizer(model, config):
    if config.training.optimizer == "mano":
        from src.utils.optimizer.mano_v2 import Mano_v2

        # Setup trainable parameters, track the inputs and output layer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        exclude_params = create_mano_exclude(model)
        exclude_params_ids = {id(p) for p in exclude_params}

        # Split up parameters for Mano (Muon) and AdamW
        mano_params = [p for p in trainable_params if p.ndim == 2 and id(p) not in exclude_params_ids]
        mano_ids = {id(p) for p in mano_params}
        adamw_params = [p for p in trainable_params if id(p) not in mano_ids]

        # Initialize the Mano Optimizer
        optimizer = Mano_v2(mano_params=mano_params, 
                            lr=config.training.learning_rate, 
                            wd=config.training.weight_decay, 
                            momentum=0.95, 
                            adamw_params=adamw_params, 
                            adamw_betas=(config.training.beta1, config.training.beta2), 
                            adamw_eps=1e-8)
        return optimizer, optimizer
    else:
        import timm.optim
        
        base_optimizer = timm.optim.create_optimizer_v2(
            model,
            opt=config.training.optimizer, # 建议这里也用 config 变量，方便切换 adamw 或 lion
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay, 
            betas=(config.training.beta1, config.training.beta2),
            filter_bias_and_bn=True
        )
        
        return base_optimizer, base_optimizer
    
def set_global_seed(seed=None):
    import os
    import random
    import secrets
    import logging
    import numpy as np
    import torch
    
    """
    设置全局随机种子。
    如果 seed 为 None，则从操作系统的 /dev/urandom 获取一个真随机数。
    """
    # 1. 生成真随机种子
    if seed is None:
        # secrets.randbits(32) 底层直接调用 os.urandom(4)
        # 生成一个 32 位的无符号整数 (0 ~ 4294967295)
        # 必须是 32 位，因为 Numpy 的 seed 最大只接受 2**32 - 1
        seed = secrets.randbits(32)
    
    # 2. 打印并记录！(极其重要：否则你跑出好结果后无法复现)
    logging.info(f"==================================================")
    logging.info(f"🌱 [Random Seed] Using True Random Seed: {seed}")
    logging.info(f"==================================================")

    # 3. Python 内置模块
    random.seed(seed)
    
    # 4. Numpy
    np.random.seed(seed)
    
    # 5. PyTorch CPU
    torch.manual_seed(seed)
    
    # 6. PyTorch GPU(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
        
    # 7. CuDNN 确定性选项 (如果需要极致的按种子复现)
    # 注意：下面两行设为 True/False 会轻微降低卷积和注意力的计算速度，
    # 但能保证在给定这个 seed 的情况下，每次跑出来的数值绝对一模一样。
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    return seed    