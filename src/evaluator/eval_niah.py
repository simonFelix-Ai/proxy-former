import logging
import os
import re
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")

from src.dataset.build import get_data_loader
from src.utils.utils import try_release_gpu_mem, handle_checkpoint, compile_model
from ..utils.setup import get_device, get_tokenizer, get_ar_model

def plot_heatmap(results, save_path="niah_heatmap.png"):
    if not results:
        logging.info("No results to plot.")
        return

    # 1. 明确定义要展示的所有测试长度（包含 1024 / 1K）
    context_lengths = [
        1024 * k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ]
    depth_percents = list(np.linspace(0.0, 1.0, 11))  # 0% - 100%

    num_depths = len(depth_percents) - 1  # 10 个深度区间 (0% - 90%)
    num_contexts = len(context_lengths)  # 11 个上下文长度

    scores = np.full((num_depths, num_contexts), np.nan)
    counts = np.zeros((num_depths, num_contexts))
    hits = np.zeros((num_depths, num_contexts))

    for r in results:
        haystack_len = r["haystack_len"]
        depth = r["depth"]
        hit = r["hit"]

        # 【修复点 1】：找到距离 haystack_len 最近或第一个 >= 的离散长度索引
        # 比如 1024 映射到 0, 2048 映射到 1
        c_idx = next(
            (i for i, v in enumerate(context_lengths) if haystack_len <= v),
            len(context_lengths) - 1,
        )

        d_idx = int(depth * 10)
        d_idx = max(0, min(d_idx, num_depths - 1))

        counts[d_idx, c_idx] += 1
        if hit:
            hits[d_idx, c_idx] += 1

    for i in range(num_depths):
        for j in range(num_contexts):
            if counts[i, j] > 0:
                scores[i, j] = hits[i, j] / counts[i, j]

    plt.figure(figsize=(max(10, len(context_lengths) * 1.2), 8))

    # 【修复点 2】：包含 1024 的完整标签
    x_labels = [
        f"{v}\n({v//1024}K)" if v < 1048576 else f"{v}\n(1M)"
        for v in context_lengths
    ]
    y_labels = [f"{v * 100:.0f}%" for v in depth_percents[:-1]]

    sns.heatmap(
        scores,
        cmap="RdYlGn",
        xticklabels=x_labels,
        yticklabels=y_labels,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Retrieval Accuracy"},
        linewidths=0.5,
    )

    plt.title("Needle In A Haystack - Passkey Retrieval Accuracy", fontsize=14)
    plt.xlabel("Context Length Bins (Tokens)", fontsize=12)
    plt.ylabel("Needle Depth Bins (% of document)", fontsize=12)
    plt.tight_layout()

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Heatmap saved to {save_path}")


def run_niah_eval(config, model, tokenizer, loader_eval, device, max_eval_batches=50):
    import time

    model.eval()
    
    pad_id = tokenizer.pad_token_id if (tokenizer and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None) else 0

    results = []
    hits = 0
    total = 0

    eval_start_time = time.time()

    progress_bar = tqdm(loader_eval, desc="Evaluating Multi-Passkey")
    for step, batch in enumerate(progress_bar):
        if step >= max_eval_batches:
            break
            
        histories = batch["history"]
        
        for b in range(histories.size(0)):
            h_tensor = histories[b]
            
            valid_h = h_tensor[h_tensor != pad_id]
            if len(valid_h) == 0:
                continue
                
            h_text = tokenizer.decode(valid_h.tolist(), skip_special_tokens=True)
            
            matches = list(re.finditer(r'The passkey_(\d+) is (\d+)', h_text))
            
            for match in matches:
                pk_id_str, pk_val = match.groups()
                depth = match.start() / max(1, len(h_text))
                
                q_text = f"\nWhat is the passkey_{pk_id_str}? The passkey_{pk_id_str} is"
                input_ids = tokenizer.encode(q_text, add_special_tokens=False)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                h_batch = histories[b].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    generated_tokens = []
                    with torch.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                        token_generator = model.generate(
                            input_tensor, max_new_tokens=10, do_sample=False,
                            history=h_batch
                        )
                    
                    for next_token in token_generator:
                        generated_tokens.append(next_token.item())
                        
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                is_hit = pk_val in generated_text
                
                results.append({
                    "haystack_len": config.data.history_len,
                    "depth": depth,
                    "hit": is_hit
                })
                
                total += 1
                if is_hit:
                    hits += 1
                    
        acc = hits / max(1, total) * 100
        progress_bar.set_postfix({"acc": f"{acc:.1f}%", "total": total})

    eval_end_time = time.time()
    print(f"eval spend time: {eval_end_time - eval_start_time}")

    return results, hits / max(1, total) if total > 0 else 0.0

def evaluate(config):
    """
    Main evaluation orchestrator for Needle In A Haystack (NIAH).
    """    
    device = get_device(config)
    
    tokenizer = get_tokenizer(config)
    
    loader_train, loader_eval = get_data_loader(config)
    if loader_eval is None:
        raise ValueError("Cannot find loader_eval!")
        
    ar_model = get_ar_model(config, device)

    ar_ckpt_path = f"out/checkpoints/train/{config.experiment_name}/ar.pth"
    handle_checkpoint(ar_model, ckpt_path=ar_ckpt_path, save=False)

    if config.hardware.compile:
        logging.info("Compiling the ar_model with torch.compile()...")
        ar_model = compile_model(ar_model, auto_eager=config.hardware.auto_eager)

    try_release_gpu_mem(device)
    
    # Run evaluation
    results, overall_acc = run_niah_eval(
        config, ar_model, tokenizer, loader_eval, device, max_eval_batches=config.eval.max_eval_batch
    )

    haystack_len = config.data.history_len
    nkey = config.data.num_passkeys
    save_path = f"out/eval/{config.experiment_name}/niah.heatmap.haystack_{haystack_len}.nkey_{nkey}.png"
    plot_heatmap(results, save_path=save_path)

    logging.info(f"\nOverall retrieval accuracy: {overall_acc * 100:.1f}%")