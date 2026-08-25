import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class ScalarEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: [B] 的浮点数 t (0~1)
        time = time * 1000.0 
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len = 100000, dropout = 0.0):
        super().__init__()
        # 1. 计算 PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 形状: (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # 2. Dropout (可选，默认为 0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, length: int) -> torch.Tensor:
        """
        只返回位置编码张量，不涉及与 Input 相加。
        Args:
            length: 需要的序列长度
        Returns:
            (1, length, d_model)
        """
        if length > self.pe.size(1):
            raise ValueError(f"Requested length {length} exceeds max_len {self.pe.size(1)}")
            
        return self.dropout(self.pe[:, :length, :])

class Rope1D(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # 核心优化：只缓存 inv_freq 这一小段常数（长度仅为 dim // 2）
        # 这点数据连 1KB 都不到，极其轻量
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, qk_pos_ids=None, offset=0):
        """
        q, k: [B, H, T, D]
        qk_pos_ids: [B, T] 或 [T]
        """
        B, H, T, D = q.shape

        # 1. 规范化 qk_pos_ids
        if qk_pos_ids is None:
            # 配合 KV Cache：从 offset 开始生成位置 ID
            qk_pos_ids = torch.arange(offset, offset + T, device=q.device).unsqueeze(0) # [1, T]
        elif qk_pos_ids.dim() == 1:
            # 如果只传了一维的 [T]，扩展为 [1, T] 方便统一处理
            qk_pos_ids = qk_pos_ids.unsqueeze(0)
            
        # 2. 转换为高精度浮点进行计算，防止千万级别整数运算溢出或精度丢失
        # 注意：float32 可以完美无损表示最大 16,777,216 的整数。
        pos = qk_pos_ids.to(torch.float32)

        # 3. 核心动态计算 (广播机制：[B, T, 1] * [dim//2] -> [B, T, dim//2])
        freqs = pos.unsqueeze(-1) * self.inv_freq
        
        # 4. 直接拼接维度对齐原始逻辑
        emb = torch.cat((freqs, freqs), dim=-1) #[B, T, D]
        
        # 5. 求三角函数并增加 H 维度以适应[B, H, T, D] -> [B, 1, T, D]
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)

        # 6. 延迟转换数据格式 (例如转换回 bfloat16 或 float16)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        # 7. 施加旋转
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

class Rope2D(nn.Module):
    """
    2D Rotary Positional Embedding
    - row uses first half of head_dim
    - col uses second half of head_dim
    """
    def __init__(self, dim, base = 10000):
        super().__init__()

        head_dim = dim
        
        assert head_dim % 2 == 0, "head_dim must be even"
        
        self.head_dim = head_dim
        self.half = head_dim // 2
        self.quarter = self.half // 2

        inv_freq = 1.0 / (
            base ** (torch.arange(self.quarter, dtype=torch.float32) / self.quarter)
        )
        # buffer: 不参与训练，自动随 device / dtype 走
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # cache
        self._cached_hw = None
        self._cached_sin = None
        self._cached_cos = None

    @torch.no_grad()
    def _build_cache(self, H: int, W: int, device, dtype):
        """
        构建 [N, half] 的 sin / cos cache
        """
        N = H * W
        row = torch.arange(H, device=device).repeat_interleave(W)
        col = torch.arange(W, device=device).repeat(H)

        inv_freq = self.inv_freq.to(device=device, dtype=dtype)

        # angles
        sin_r = torch.sin(row[:, None] * inv_freq[None, :])
        cos_r = torch.cos(row[:, None] * inv_freq[None, :])
        sin_c = torch.sin(col[:, None] * inv_freq[None, :])
        cos_c = torch.cos(col[:, None] * inv_freq[None, :])

        # expand to match half-dim
        sin_r = sin_r.repeat_interleave(2, dim=-1)
        cos_r = cos_r.repeat_interleave(2, dim=-1)
        sin_c = sin_c.repeat_interleave(2, dim=-1)
        cos_c = cos_c.repeat_interleave(2, dim=-1)

        # concat row + col
        sin = torch.cat([sin_r, sin_c], dim=-1)  # [N, D]
        cos = torch.cat([cos_r, cos_c], dim=-1)

        self._cached_hw = (H, W)
        self._cached_sin = sin
        self._cached_cos = cos

    def rotate_half(self, x):
        # x: [..., D]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q, k, H: int, W: int):
        """
        q, k: [B, heads, N, head_dim]
        """
        B, heads, N, D = q.shape
        assert D == self.head_dim
        assert N == H * W

        device = q.device
        dtype = q.dtype

        # build cache if needed
        if (
            self._cached_hw != (H, W)
            or self._cached_sin.device != device
            or self._cached_sin.dtype != dtype
        ):
            self._build_cache(H, W, device, dtype)

        sin = self._cached_sin  # [N, D]
        cos = self._cached_cos

        # reshape for broadcast
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]

        # apply rope
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)

        return q, k
