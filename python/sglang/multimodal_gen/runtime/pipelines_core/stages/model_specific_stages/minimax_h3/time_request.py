# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    penultimate_sigma: float | None = None,
    two_stage: bool | None = None,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import os
    import torch

    # 判定是否属于两阶段（二采）模式
    if two_stage is None:
        env_val = os.environ.get("AGNES_TWO_STAGE_SAMPLING", "").strip().lower()
        if env_val in {"1", "true", "yes", "on"} and num_steps in (8, 12):
            is_two_stage = True
        else:
            is_two_stage = False
    else:
        is_two_stage = bool(two_stage)

    # 1. 两阶段 (二采) 模式:
    # 均加入产线级 0.2 -> 0.0 收尾过渡步：
    # 4+4 模式: 一采 4 步 + 二采 5 步 (4 步高清精修 + 1 步 0.2->0.0 收尾) = 共 9 步去噪 (10 个节点)
    # 8+4 模式: 一采 8 步 + 二采 5 步 (4 步高清精修 + 1 步 0.2->0.0 收尾) = 共 13 步去噪 (14 个节点)
    if is_two_stage:
        if penultimate_sigma is not None:
            pen_sigma = float(penultimate_sigma)
        else:
            env_val = os.environ.get("SGLANG_H3_8N1_SIGMA")
            if env_val:
                try:
                    pen_sigma = float(env_val)
                except ValueError:
                    env_val = None
            if not env_val:
                base_50 = torch.linspace(1.0, 0.0, 50, device="cpu", dtype=torch.float32)
                shift_50 = float(shift_scale) * base_50 / (1 + (float(shift_scale) - 1) * base_50)
                pen_sigma = float(shift_50[48].item())

        base = torch.linspace(
            1.0,
            0.0,
            int(num_steps) + 1,
            device="cpu",
            dtype=torch.float32,
        )
        shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
        sigmas_base = [float(v) for v in shifted.tolist()]
        # 截断最后一个 0.0，拼入 [0.2, 0.0] 作为末尾收尾过渡步
        return sigmas_base[:-1] + [pen_sigma, 0.0]

    # 2. 产线单阶段直推标准 8+1 调度:
    # 当 SGLANG_H3_8N1_MODE == "1" 或 num_steps in (9, 10) 时激活，生成 10 个节点，去噪严格执行 9 steps (8 步常规 + 1 步 0.2->0.0 收尾)
    is_8n1 = (os.environ.get("SGLANG_H3_8N1_MODE") == "1" or num_steps in (9, 10))
    if is_8n1 and num_steps > 1:
        # (1) 确定倒数第二步节点的 sigma 强度: 视频 (shift_scale ~ 12.0) 对应 0.2; 音频 (shift_scale ~ 3.0) 对应 ~0.05882
        if penultimate_sigma is not None:
            pen_sigma = float(penultimate_sigma)
        else:
            env_val = os.environ.get("SGLANG_H3_8N1_SIGMA")
            if env_val:
                try:
                    pen_sigma = float(env_val)
                except ValueError:
                    env_val = None
            if not env_val:
                base_50 = torch.linspace(1.0, 0.0, 50, device="cpu", dtype=torch.float32)
                shift_50 = float(shift_scale) * base_50 / (1 + (float(shift_scale) - 1) * base_50)
                pen_sigma = float(shift_50[48].item())

        # (2) 计算前置基础线性调度:
        # 当 num_steps 为 9 或 10 时，base_steps 固定为 8，产生 8 个前置点 + 2 个收尾点 [pen_sigma, 0.0] = 共 10 个节点 (9 步去噪)
        base_steps = 8 if num_steps in (9, 10) else (num_steps - 1)
        base = torch.linspace(1.0, 0.0, base_steps + 1, device="cpu", dtype=torch.float32)
        shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
        sigmas_base = [float(v) for v in shifted.tolist()]

        # (3) 拼接收尾节点: 前 base_steps 个节点 + [pen_sigma, 0.0] -> 精准 10 个节点，SGLang 迭代 len-1 = 9 步去噪
        return sigmas_base[:-1] + [pen_sigma, 0.0]

    # 3. 其它情况下的标准 Rectified Flow 调度:
    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps),
        device="cpu",
        dtype=torch.float32,
    )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]

