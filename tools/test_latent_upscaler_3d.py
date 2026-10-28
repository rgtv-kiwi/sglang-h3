#!/usr/bin/env python3
import importlib.util
import os
import sys
import time
from pathlib import Path
import torch

# 1. 动态加载 MiniMax-H3 3D Latent Upscaler 模块
MODULE_PATH = Path(__file__).resolve().parent.parent / "python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/latent_upscaler_3d.py"
spec = importlib.util.spec_from_file_location("latent_upscaler_3d", str(MODULE_PATH))
upscaler_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upscaler_mod)

load_minimax_h3_latent_upscaler_3d = upscaler_mod.load_minimax_h3_latent_upscaler_3d
upscale_minimax_h3_video_latent = upscaler_mod.upscale_minimax_h3_video_latent

WEIGHTS_PATH = upscaler_mod.DEFAULT_LATENT_UPSCALER_PATH

def main():
    print("=" * 70)
    print("🚀 MiniMax-H3 3D Latent Upscaler 模块验证测试 (sglang-fork)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 运行设备: {device}")
    if device == "cuda":
        print(f"[*] GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"[*] 当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # 1. 加载权重
    print(f"\n[1/3] 加载 3D 卷积超分权重: {WEIGHTS_PATH}")
    t0 = time.time()
    model = load_minimax_h3_latent_upscaler_3d(
        weights_path=WEIGHTS_PATH,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    t_load = time.time() - t0
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ 加载成功! 参数量: {params:,} ({params * 2 / 1024**2:.1f} MB FP16), 耗时: {t_load:.2f}s")
    if device == "cuda":
        print(f"[*] 模型加载后显存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # 2. 构造一采（低分辨率 Draft 0.2MP）合成 Latent
    # 0.2MP 在 16:9 下约为 608x352 像素
    # 对应的 Latent 尺寸: (H_in = 352/16 = 22, W_in = 608/16 = 38)
    # 帧数: 17 帧 (MiniMax-H3 典型块)
    B, C, T, H_in, W_in = 1, 24, 17, 22, 38
    print(f"\n[2/3] 构建一采 Draft Latent: shape=[B={B}, C={C}, T={T}, H={H_in}, W={W_in}]")
    print(f"[*] 对应像素分辨率: {W_in * 16}x{H_in * 16} (约 0.2MP)")

    torch.manual_seed(2026)
    draft_latent = torch.randn(
        B, C, T, H_in, W_in,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # 3. 执行 3D 空间潜空间超分
    print(f"\n[3/3] 执行 2.0x 3D 潜空间超分辨率放大...")
    t1 = time.time()
    upscaled_latent = upscale_minimax_h3_video_latent(
        latent=draft_latent,
        model=model,
        scale=2.0,
        align=32,
        enable_temporal_chunking=True,
        device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t_infer = time.time() - t1

    B_out, C_out, T_out, H_out, W_out = upscaled_latent.shape
    print(f"✅ 超分执行完成! 耗时: {t_infer:.3f}s")
    print(f"[*] 输出 Latent shape: [B={B_out}, C={C_out}, T={T_out}, H={H_out}, W={W_out}]")
    print(f"[*] 对应目标像素分辨率: {W_out * 16}x{H_out * 16} (~720P)")

    # 4. 关键指标校验
    assert B_out == B and C_out == C and T_out == T, f"维度不匹配: expected ({B},{C},{T}), got ({B_out},{C_out},{T_out})"
    assert H_out == H_in * 2 and W_out == W_in * 2, f"放大倍率异常: expected ({H_in*2},{W_in*2}), got ({H_out},{W_out})"
    assert not torch.isnan(upscaled_latent).any(), "输出包含 NaN 异常值!"
    assert not torch.isinf(upscaled_latent).any(), "输出包含 Inf 异常值!"

    print("\n" + "=" * 70)
    print("🎉 单元测试全部通过 (ALL CHECKS PASSED)！")
    print(f"   输入: {W_in*16}x{H_in*16} ({H_in}x{W_in} latent) -> 输出: {W_out*16}x{H_out*16} ({H_out}x{W_out} latent)")
    print(f"   耗时: {t_infer:.3f}s, 数值范围: [{upscaled_latent.min():.2f}, {upscaled_latent.max():.2f}]")
    print("=" * 70)

if __name__ == "__main__":
    main()
