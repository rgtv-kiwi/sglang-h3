#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test Step 4: Two-stage denoise loop and transition state machine.

Validates:
1. TwoStageDenoiseManager: Token unpatchify -> 3D Latent Upscaler -> Token patchify
2. Layout reconstruction and RoPE cache rebuild
3. Full 8-step cascade denoise loop execution with transition at step 4 (split_step=4)
4. Verification of video rows count change (3553 -> 14212)
"""

import sys
import os
import time
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.latent_upscaler_3d import (
    TwoStageDenoiseManager,
    DEFAULT_LATENT_UPSCALER_PATH,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)


def test_upscale_and_repack():
    print("=" * 70)
    print("TEST 1: TwoStageDenoiseManager Token Repack Test")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manager = TwoStageDenoiseManager(
        enabled=True,
        split_step=4,
        scale_factor=2.0,
        model_path=DEFAULT_LATENT_UPSCALER_PATH,
        device=device,
        dtype=torch.float16,
    )

    latent_t = 17
    # 0.2MP draft: 352x608 -> latent 22x38 -> patchified rows: 17 * (22//2) * (38//2) = 17 * 11 * 19 = 3553
    draft_h, draft_w = 22, 38
    num_rows = latent_t * (draft_h // 2) * (draft_w // 2)
    fake_token_rows = torch.randn(num_rows, 96, dtype=torch.float32, device=device)

    print(f"Input draft rows: {fake_token_rows.shape} for (T={latent_t}, H={draft_h}, W={draft_w})")
    start = time.perf_counter()
    new_rows, new_h, new_w = manager.upscale_and_repack(
        fake_token_rows,
        latent_t=latent_t,
        latent_h=draft_h,
        latent_w=draft_w,
    )
    elapsed = time.perf_counter() - start
    print(f"Output high-res rows: {new_rows.shape} for (T={latent_t}, H={new_h}, W={new_w}) in {elapsed:.3f}s")

    expected_h = 44
    expected_w = 76
    expected_rows = latent_t * (expected_h // 2) * (expected_w // 2)  # 17 * 22 * 38 = 14212

    assert new_h == expected_h, f"Expected new_h={expected_h}, got {new_h}"
    assert new_w == expected_w, f"Expected new_w={expected_w}, got {new_w}"
    assert new_rows.shape == (expected_rows, 96), f"Expected shape {(expected_rows, 96)}, got {new_rows.shape}"
    assert not torch.isnan(new_rows).any(), "NaN detected in output rows!"
    print(">>> TEST 1 PASSED: Token Repack successfully scaled 4x tokens (2x spatial)!\n")


class MockH3DiT(nn.Module):
    """Mock H3 DiT for testing denoise loop transitions."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def prepare_adaln_plans(self, unique_timesteps_list):
        return [None] * len(unique_timesteps_list)

    def build_rope_cache(self, img_position_ids, device):
        return torch.zeros((img_position_ids.shape[0], 64), device=device)

    def forward(self, **kwargs):
        infer_out_pos = kwargs.get("img_pos_for_infer_output_info")
        audio_pos = kwargs.get("audio_pos_info")
        x = kwargs["x"]
        audio_x = kwargs["audio_x"]
        device = x.device
        dtype = x.dtype

        if isinstance(infer_out_pos, dict):
            infer_out_pos = infer_out_pos["position_ids"]
        if isinstance(audio_pos, dict):
            audio_pos = audio_pos["position_ids"]

        n_video = infer_out_pos.shape[0] if infer_out_pos is not None else x.shape[1]
        n_audio = audio_pos.shape[0] if audio_pos is not None else audio_x.shape[1]

        v_video = torch.zeros(n_video, 96, dtype=dtype, device=device)
        v_audio = torch.zeros(n_audio, 32, dtype=dtype, device=device)
        return v_video, v_audio


def test_two_stage_denoise_loop():
    print("=" * 70)
    print("TEST 2: Full 8-Step Denoise Loop with Two-Stage Transition")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup schedules: 9 sigmas -> 8 steps (intervals)
    sigmas_video = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=6.0)
    sigmas_audio = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=6.0)
    num_steps = len(sigmas_video) - 1
    print(f"Num steps: {num_steps}, Sigmas video: {[round(s, 4) for s in sigmas_video]}")

    text_len = 16
    latent_t = 17
    audio_t = 40  # 1 second audio
    draft_h, draft_w = 22, 38
    final_h, final_w = 44, 76

    # 2. Build initial draft packed layout
    draft_packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=draft_h,
        latent_w=draft_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    text_embeddings = torch.randn(text_len, 4096, dtype=torch.float32, device=device)
    tags = draft_packed["token_tags"].to(device)

    positive = MiniMaxH3DenoiseBranch(
        packed=draft_packed,
        text_embeddings=text_embeddings,
        token_tags=tags,
        device=device,
    )

    model = MockH3DiT().to(device)
    positive.static_kwargs["rope_cache"] = model.build_rope_cache(
        positive.static_kwargs["img_position_ids"], device=device
    )

    # Initial random noise
    n_draft_video = positive.img_pos.shape[0]
    n_audio = positive.audio_pos.shape[0]
    initial_video = torch.randn(int(n_draft_video), 96, dtype=torch.float32, device=device)
    initial_audio = torch.randn(int(n_audio), 32, dtype=torch.float32, device=device)

    print(f"Initial video rows: {initial_video.shape}, audio rows: {initial_audio.shape}")

    # Manager
    manager = TwoStageDenoiseManager(
        enabled=True,
        split_step=4,
        scale_factor=2.0,
        model_path=DEFAULT_LATENT_UPSCALER_PATH,
        device=device,
        dtype=torch.float16,
    )

    # Rebuild layout callback
    def rebuild_layout_fn(new_h: int, new_w: int):
        print(f"  --> Rebuild layout callback invoked for ({new_h}, {new_w})")
        new_packed = minimax_h3_packed_sequence(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=new_h,
            latent_w=new_w,
            audio_t=audio_t,
            include_keyframe_cond=False,
        )
        new_tags = new_packed["token_tags"].to(device)
        new_pos = MiniMaxH3DenoiseBranch(
            packed=new_packed,
            text_embeddings=text_embeddings,
            token_tags=new_tags,
            device=device,
        )
        new_pos.static_kwargs["rope_cache"] = model.build_rope_cache(
            new_pos.static_kwargs["img_position_ids"], device=device
        )
        return new_pos, None

    step_history = []
    def on_step(step, v_rows, a_rows):
        step_history.append((step, v_rows.shape[0]))
        print(f"  Step {step}: video_rows = {v_rows.shape[0]}, audio_rows = {a_rows.shape[0]}")

    loop_output_state = {}
    final_video, final_audio = minimax_h3_denoise_loop(
        model=model,
        positive=positive,
        initial_video_rows=initial_video,
        initial_audio_rows=initial_audio,
        keyframe_cond_rows=None,
        sigmas_video=sigmas_video,
        sigmas_audio=sigmas_audio,
        device=device,
        on_step=on_step,
        two_stage_manager=manager,
        rebuild_layout_fn=rebuild_layout_fn,
        current_latent_dims=(latent_t, draft_h, draft_w),
        output_state=loop_output_state,
    )

    print(f"\nDenoise Loop Finished!")
    print(f"Final video rows: {final_video.shape}")
    print(f"Final audio rows: {final_audio.shape}")
    print(f"Output state: {loop_output_state}")

    # Check step history: steps 0..3 have 3553 video rows; steps 4..7 have 14212 video rows!
    for step, rows in step_history:
        if step < 4:
            assert rows == 3553, f"Step {step} should have 3553 rows, got {rows}"
        else:
            assert rows == 14212, f"Step {step} should have 14212 rows, got {rows}"

    assert final_video.shape[0] == 14212, f"Final video rows should be 14212, got {final_video.shape[0]}"
    assert loop_output_state["latent_h"] == 44, f"Output latent_h should be 44, got {loop_output_state['latent_h']}"
    assert loop_output_state["latent_w"] == 76, f"Output latent_w should be 76, got {loop_output_state['latent_w']}"

    print(">>> TEST 2 PASSED: 8-Step Cascade Denoise Loop seamlessly executed stage 1 and stage 2!\n")


if __name__ == "__main__":
    test_upscale_and_repack()
    test_two_stage_denoise_loop()
    print("=" * 70)
    print("ALL STEP 4 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
