# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 3D Latent Upscaler for two-stage (cascade) video generation.

Pure PyTorch implementation of the 3D convolutional latent upscaling network.
Extracts and optimizes the core architecture from ComfyUI for direct, high-performance
integration with SGLang Diffusion runtime.
"""

from __future__ import annotations

import gc
import os
import re
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file

# ==============================================================================
# MiniMax H3 潜空间归一化统计量 (24 个通道)
# ==============================================================================
# MiniMax-H3 的 24 通道 3D VAE 潜空间具有非零均值和非单位方差分布。
# 在将潜空间送入 3D 卷积超分网络 (LatentResizer3D) 之前，必须执行标准归一化 (z = (x - mean) / std)，
# 超分完成后执行逆归一化 (x = z * std + mean)，以保证数值稳定性并避免画面偏色或对比度坍塌。
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

# VAE 下采样倍率：像素空间相对于潜空间的缩小倍数 (16x)
VAE_DOWNSAMPLE = 16


def _make_norm_tensors(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造广播形状为 [1, 24, 1, 1, 1] 的均值与方差归一化张量。"""
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


# ==============================================================================
# 3D 卷积超分基础模块 (GroupNorm, 3D Attention, 3D 残差块, 时序卷积)
# ==============================================================================
def normalization(channels: int) -> nn.GroupNorm:
    """标准 32-Group GroupNorm，适配 3D 卷积特征通道归一化。"""
    return nn.GroupNorm(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0, out_channels: Optional[int] = None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


# ==============================================================================
# 纯 3D 潜空间超分骨干网络 (LatentResizer3D)
# ==============================================================================
# 核心结构特点：
# 1. 采用时空 3D 卷积体系，通道数 512，输入输出 24 通道；
# 2. 结合三线性插值 (Trilinear Interpolation) 实现空间平滑放大；
# 3. 采用分段/时序窗口分块 (Temporal Chunking) 机制，支持任意长序列并防止显存瞬时峰值 OOM。
class LatentResizer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        dropout: float = 0.1,
        attn: bool = False,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[float] = None,
        target_size: Optional[Tuple[int, int, int]] = None,
        enable_chunking: bool = True,
    ) -> torch.Tensor:
        """3D 超分前向计算，支持重叠时序分块 (Temporal Overlap Chunking) 以抑制显存爆炸。
        
        Args:
            x: 5D 输入潜空间张量 [B, C, T, H, W]
            scale: 缩放因子 (例如 2.0 代表空间尺寸放大 2 倍)
            target_size: 目标输出尺寸 (T, H_out, W_out)，优先于 scale
            enable_chunking: 是否开启时序分块融合推理 (长视频时防止 OOM)
        """
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        tk = 0
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break

        overlap = tk if tk > 0 else 5
        chunk = 32

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        # Temporal chunking with blended overlapping windows
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk
            del seg, seg_out, valid_out

        out_full = out_full / weight_full.clamp(min=1e-8)
        return out_full

    def _forward_seg(
        self,
        x: torch.Tensor,
        scale: Optional[float],
        size: Tuple[int, int, int],
    ) -> torch.Tensor:
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# ==============================================================================
# Model Loading & Architecture Auto-detection
# ==============================================================================
def _detect_arch(sd: dict[str, torch.Tensor]) -> dict:
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = "conv_in.weight"
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r"in_blocks\.(\d+)\.dwconv\.weight", k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.dwconv\.weight", k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if "dwconv.weight" in k and k.endswith("dwconv.weight"):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    cfg["attn"] = any("attn" in k for k in sd)
    # The official 3D upscaler checkpoint does not use self-attention to maintain low VRAM
    cfg["attn"] = False
    return cfg


def load_minimax_h3_latent_upscaler_3d(
    weights_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> LatentResizer3D:
    """Load MiniMax H3 3D Latent Upscaler from a safetensors file."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"3D Latent Upscaler weights not found: {weights_path}")

    if weights_path.endswith(".pth") or weights_path.endswith(".pt"):
        raw_sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
            raw_sd = raw_sd["state_dict"]
    else:
        raw_sd = load_file(weights_path, device="cpu")
    # Strip any potential prefix
    sd = {
        (k[len("upscaler."):] if k.startswith("upscaler.") else k): v
        for k, v in raw_sd.items()
    }
    cfg = _detect_arch(sd)

    model = LatentResizer3D(
        in_channels=cfg["in_channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        channels=cfg["channels"],
        dropout=cfg["dropout"],
        attn=cfg["attn"],
        temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model


# ==============================================================================
# 端到端 3D 潜空间超分执行函数
# ==============================================================================
def upscale_minimax_h3_video_latent(
    latent: torch.Tensor,
    model: LatentResizer3D,
    scale: float = 2.0,
    align: int = 32,
    enable_temporal_chunking: bool = True,
    device: str = "cuda",
    target_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """对视频潜空间 [B, 24, T, H, W] 执行 3D 卷积超分（默认放大 2.0 倍）。
    
    计算管线流程：
    1. 维度校验与 5D 对齐；
    2. 像素空间 32-pixel 网格边界对齐（防止 VAE 解码时出现非整除黑边）；
    3. H3 统计量减均值、除方差标准化；
    4. 调用 LatentResizer3D 执行 3D 卷积与三线性空间放大；
    5. 乘方差、加均值反归一化还原；
    6. 恢复原始张量数据类型返回。
    """
    orig_dtype = latent.dtype
    target_device = torch.device(device)
    compute_dtype = next(model.parameters()).dtype

    # Ensure 5D [B, C, T, H, W]
    was_4d = (latent.dim() == 4)
    s = latent.unsqueeze(2) if was_4d else latent
    s = s.to(device=target_device, dtype=compute_dtype, copy=True)

    b, c, t, h_in, w_in = s.shape

    # 1. 像素空间对齐 (MiniMax-H3 VAE 空间下采样为 16x，保证放大后像素为 32 的整数倍)
    downsample = VAE_DOWNSAMPLE
    w_pixel_target = w_in * downsample * scale
    h_pixel_target = h_in * downsample * scale

    w_pixel_aligned = round(w_pixel_target / align) * align
    h_pixel_aligned = round(h_pixel_target / align) * align

    w_pixel_final = round(w_pixel_aligned / downsample) * downsample
    h_pixel_final = round(h_pixel_aligned / downsample) * downsample

    w_out = max(1, int(w_pixel_final // downsample))
    h_out = max(1, int(h_pixel_final // downsample))
    if target_hw is not None:
        # 请求画布尺寸优先：直接严格对齐最终请求潜空间分辨率（如 1376x768 对应 86x48），避免微小像素偏差
        h_out = int(target_hw[0])
        w_out = int(target_hw[1])
        if h_out <= 0 or w_out <= 0 or h_out % 2 or w_out % 2:
            raise ValueError(
                "two-stage target latent size must be positive and even, "
                f"got {h_out}x{w_out}"
            )

    # 2. Normalize with H3 latent statistics
    norm_mean, norm_std = _make_norm_tensors(target_device, compute_dtype)

    with torch.inference_mode():
        s_norm = (s - norm_mean) / norm_std
        del s
        out = model(
            s_norm,
            scale=scale,
            target_size=(t, h_out, w_out),
            enable_chunking=enable_temporal_chunking,
        )
        del s_norm
        out = out * norm_std + norm_mean

    if was_4d:
        out = out.squeeze(2)

    return out.to(dtype=orig_dtype)


# ==============================================================================
# Two-Stage Denoise Manager
# ==============================================================================
_UPSCALER_RELATIVE_PATH = (
    Path("新models")
    / "latent_upscaler"
    / "minimax_h3_latent_upscaler_3d_fp16.safetensors"
)


def _default_latent_upscaler_path() -> str:
    """Find this checkout's bundled upscaler without hard-coding its location."""
    configured_root = os.environ.get("MINIMAX_H3_PROJECT_ROOT")
    if configured_root:
        return str(Path(configured_root).expanduser().resolve() / _UPSCALER_RELATIVE_PATH)

    for parent in Path(__file__).resolve().parents:
        candidate = parent / _UPSCALER_RELATIVE_PATH
        if candidate.is_file():
            return str(candidate)

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "sglang").is_dir():
            candidate = parent / _UPSCALER_RELATIVE_PATH
            if candidate.is_file():
                return str(candidate)

    # Check well-known system mount path
    system_candidate = Path("/mnt2/comfyui-minimax-H3/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors")
    if system_candidate.is_file():
        return str(system_candidate)

    # This fallback keeps the error actionable for source trees with a nonstandard
    # layout while avoiding a dependency on a sibling checkout.
    return str(Path.cwd() / _UPSCALER_RELATIVE_PATH)


DEFAULT_LATENT_UPSCALER_PATH = _default_latent_upscaler_path()


def is_two_stage_enabled(
    batch: Any = None,
    server_args: Any = None,
    sampling_params: Any = None,
) -> bool:
    """Check whether two-stage sampling is enabled via params, extra, or env var."""
    # 1. Check sampling_params
    sp = sampling_params
    if sp is None and batch is not None:
        sp = getattr(batch, "sampling_params", None)
    if sp is not None:
        if hasattr(sp, "two_stage_sampling"):
            if not getattr(sp, "two_stage_sampling", False):
                return False
            return True
        if hasattr(sp, "extra") and isinstance(sp.extra, dict):
            if "two_stage_sampling" in sp.extra:
                return bool(sp.extra.get("two_stage_sampling"))

    # 2. Check batch.extra
    if batch is not None and hasattr(batch, "extra") and isinstance(batch.extra, dict):
        if "two_stage_sampling" in batch.extra:
            return bool(batch.extra.get("two_stage_sampling"))

    # 3. Check server_args
    if server_args is not None and hasattr(server_args, "two_stage_sampling"):
        if getattr(server_args, "two_stage_sampling", False):
            return True

    # 4. Check environment variable (仅当请求未指定时生效)
    env_val = os.environ.get("MINIMAX_H3_TWO_STAGE", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True

    return False


class TwoStageDenoiseManager:
    """两阶段（二采）去噪管理器：调度 Stage 1 (Draft) 与 Stage 2 (Refine) 之间的 3D 潜空间超分。

    职责：
    1. 惰性加载 3D Latent Upscaler 权重；
    2. 管理切换步数 (split_step) 与缩放倍率 (scale_factor)；
    3. 执行“1D Token 序列反切块 -> 5D 潜空间 3D 超分 -> 1D Token 重新切块”的全流程转换。
    """

    _model_cache: dict[tuple[str, str, torch.dtype], LatentResizer3D] = {}
    _model_cache_lock = threading.RLock()

    def __init__(
        self,
        enabled: bool = True,
        split_step: int = 4,
        scale_factor: float = 2.0,
        model_path: str = DEFAULT_LATENT_UPSCALER_PATH,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.enabled = enabled
        self.split_step = split_step
        self.scale_factor = scale_factor
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self._upscaler_model: LatentResizer3D | None = None

    def get_model(self) -> LatentResizer3D:
        """Get a process-wide cached 3D Latent Upscaler instance."""
        if self._upscaler_model is None:
            resolved_path = os.path.realpath(os.path.expanduser(self.model_path))
            if not os.path.isfile(resolved_path):
                raise FileNotFoundError(
                    "3D Latent Upscaler model not found at "
                    f"{resolved_path}. Set two_stage_upscaler_path or "
                    "MINIMAX_H3_UPSCALER_PATH to an existing safetensors file."
                )
            cache_key = (resolved_path, str(self.device), self.dtype)
            with self._model_cache_lock:
                cached = self._model_cache.get(cache_key)
                if cached is None:
                    cached = load_minimax_h3_latent_upscaler_3d(
                        resolved_path,
                        device=str(self.device),
                        dtype=self.dtype,
                    )
                    self._model_cache[cache_key] = cached
                self._upscaler_model = cached
        return self._upscaler_model

    @classmethod
    def clear_model_cache(cls) -> None:
        """Release cached upscalers, primarily for controlled service shutdown."""
        with cls._model_cache_lock:
            cls._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def upscale_and_repack(
        self,
        video_target_rows: torch.Tensor,
        *,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        target_latent_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        """将 Stage 1 预测出的视频纯净 Token 行反解为 5D 潜空间，经 3D 超分后再重新 Patchify。

        执行流：
        1. minimax_h3_unpatchify_video_tokens: [Rows, 96] -> [1, 24, T, H, W]
        2. upscale_minimax_h3_video_latent: 空间 2.0x 放大 -> [1, 24, T, 2H, 2W]
        3. minimax_h3_patchify_video_latent: [1, 24, T, 2H, 2W] -> [4*Rows, 96]

        Returns:
            (new_video_target_rows, new_latent_h, new_latent_w)
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
            minimax_h3_patchify_video_latent,
            minimax_h3_unpatchify_video_tokens,
        )

        model = self.get_model()
        # Shape: [1, 24, latent_t, latent_h, latent_w]
        video_latent = minimax_h3_unpatchify_video_tokens(
            video_target_rows,
            latent_shape=[latent_t, latent_h // 2, latent_w // 2, 24],
            patch_size=[1, 2, 2],
        )

        upscaled_latent = upscale_minimax_h3_video_latent(
            video_latent,
            model=model,
            scale=self.scale_factor,
            device=str(self.device),
            target_hw=target_latent_hw,
        )

        # Output shape: [1, 24, latent_t, new_latent_h, new_latent_w]
        _, _, _, new_h, new_w = upscaled_latent.shape
        new_video_target_rows = minimax_h3_patchify_video_latent(
            upscaled_latent,
            patch_size=[1, 2, 2],
        ).to(dtype=video_target_rows.dtype, device=video_target_rows.device)

        return new_video_target_rows, new_h, new_w


# Backward compatibility alias
load_latent_resizer_3d = load_minimax_h3_latent_upscaler_3d

__all__ = [
    "DEFAULT_LATENT_UPSCALER_PATH",
    "LatentResizer3D",
    "TwoStageDenoiseManager",
    "is_two_stage_enabled",
    "load_latent_resizer_3d",
    "load_minimax_h3_latent_upscaler_3d",
    "upscale_minimax_h3_video_latent",
]



