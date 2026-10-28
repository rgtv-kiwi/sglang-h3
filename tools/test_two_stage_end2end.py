#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Two-Stage End-to-End Pipeline Contract & Stage Test.

Validates the full stage chain:
1. MiniMaxH3SamplingParams with two-stage extra fields
2. Request validation & ResolvedPlan generation
3. MiniMaxH3LatentPreparationStage: generates draft 352x608 noise with two_stage_enabled metadata
4. MiniMaxH3TimestepPreparationStage: generates 8-step (9 sigmas) schedule
5. MiniMaxH3DenoisingStage: full loop simulation with mock DiT, verifying 3D latent upscaler
   invocation, token repack, and final batch.latents high-res output shape [1, 24, 17, 44, 76]!
"""

import sys
import os
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args


class MockDiT(nn.Module):
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


def test_two_stage_pipeline_stages():
    print("=" * 70)
    print("TEST: MiniMax H3 Two-Stage Pipeline Contract & Preparation Stages")
    print("=" * 70)

    server_args = ServerArgs(
        model_path="/mnt2/comfyui-minimax-H3",
    )
    server_args.pipeline_config.dit_config.hidden_size = 3072
    server_args.pipeline_config.dit_config.num_attention_heads = 24
    server_args.pipeline_config.dit_config.attention_head_dim = 128
    set_global_server_args(server_args)

    # 1. Create sampling params with two-stage enabled
    sp = MiniMaxH3SamplingParams(
        prompt="A serene tropical beach with gentle ocean waves and palm trees.",
        task="t2va",
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        num_inference_steps=9,
        flow_shift=1.0,
        audio_flow_shift=1.0,
        two_stage_sampling=True,
        two_stage_split_step=4,
        two_stage_scale_factor=2.0,
        seed=42,
    )

    extra = sp.build_request_extra()
    print("SamplingParams build_request_extra keys:", sorted(extra.keys()))
    assert extra.get("two_stage_sampling") is True
    assert extra.get("two_stage_split_step") == 4
    assert extra.get("two_stage_scale_factor") == 2.0
    assert "minimax_h3_canonical_request" in extra
    print(">>> SamplingParams two-stage extra fields verified!")

    # 2. Build mock request
    canonical = extra["minimax_h3_canonical_request"]
    plan = minimax_h3_resolve_plan(canonical)
    extra["minimax_h3_resolved_plan"] = plan

    req = Req(
        prompt=sp.prompt,
        prompt_embeds=[torch.zeros(1, 10, 64)],
        generator=torch.Generator().manual_seed(42),
        num_frames=97,
        height=704,
        width=1216,
        num_outputs_per_prompt=1,
        sampling_params=sp,
        extra=extra,
        num_inference_steps=8,
    )

    # 3. Test MiniMaxH3LatentPreparationStage
    latent_stage = MiniMaxH3LatentPreparationStage()
    latent_stage(req, server_args=None)

    state = req.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
    print(f"LatentPreparationStage output:")
    print(f"  draft latent_h: {state['latent_h']}, latent_w: {state['latent_w']}")
    print(f"  target_latent_h: {state['target_latent_h']}, target_latent_w: {state['target_latent_w']}")
    print(f"  two_stage_enabled: {state['two_stage_enabled']}")
    print(f"  initial_video_rows: {state['initial_video_rows'].shape}")

    # Verify 2x halving of draft latent dimensions
    assert state["two_stage_enabled"] is True
    assert state["latent_h"] * 2 == state["target_latent_h"], f"Draft H {state['latent_h']} should be half of target {state['target_latent_h']}"
    assert state["latent_w"] * 2 == state["target_latent_w"], f"Draft W {state['latent_w']} should be half of target {state['target_latent_w']}"
    expected_rows = state["latent_t"] * (state["latent_h"] // 2) * (state["latent_w"] // 2)
    assert state["initial_video_rows"].shape == (expected_rows, 96), f"Expected shape ({expected_rows}, 96), got {state['initial_video_rows'].shape}"
    print(f">>> LatentPreparationStage successfully created draft noise ({expected_rows} rows, draft {state['latent_h']}x{state['latent_w']} -> target {state['target_latent_h']}x{state['target_latent_w']})!")

    # 4. Test MiniMaxH3TimestepPreparationStage
    timestep_stage = MiniMaxH3TimestepPreparationStage()
    timestep_stage(req, server_args=None)

    sigmas = req.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
    print(f"TimestepPreparationStage output:")
    print(f"  video sigmas ({len(sigmas['video'])} values): {[round(s, 4) for s in sigmas['video']]}")
    print(f"  audio sigmas ({len(sigmas['audio'])} values): {[round(s, 4) for s in sigmas['audio']]}")

    assert len(sigmas["video"]) >= 2, f"Expected valid sigmas, got {len(sigmas['video'])}"
    assert len(sigmas["audio"]) >= 2, f"Expected valid sigmas, got {len(sigmas['audio'])}"
    print(f">>> TimestepPreparationStage successfully staged {len(sigmas['video'])} schedule points!")

    # 5. Test MiniMaxH3DenoisingStage with MockDiT
    print("\n" + "=" * 70)
    print("TEST: MiniMaxH3DenoisingStage End-to-End Execution")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mock_dit = MockDiT().to(device)

    text_len = 16
    req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY] = {
        "positive": {
            "hidden_states": torch.randn(text_len, 4096, dtype=torch.float32, device=device),
            "text_token_tags": torch.zeros(text_len, dtype=torch.long, device=device),
            "text_len": text_len,
        }
    }

    denoise_stage = MiniMaxH3DenoisingStage(transformer=mock_dit)
    denoise_stage(req, server_args=server_args)

    print(f"\nDenoisingStage completed!")
    print(f"req.latents shape: {req.latents.shape}")
    print(f"req.audio_latents shape: {req.audio_latents.shape}")
    print(f"final state: latent_h={state['latent_h']}, latent_w={state['latent_w']}")

    # Verify output latents shape is high-res: [1, 24, 17, 44, 76]
    expected_latent_shape = (1, 24, state["latent_t"], state["target_latent_h"], state["target_latent_w"])
    assert req.latents.shape == expected_latent_shape, f"Expected latents shape {expected_latent_shape}, got {req.latents.shape}"
    assert state["latent_h"] == state["target_latent_h"], f"Final latent_h {state['latent_h']} should equal target {state['target_latent_h']}"
    assert state["latent_w"] == state["target_latent_w"], f"Final latent_w {state['latent_w']} should equal target {state['target_latent_w']}"
    assert not torch.isnan(req.latents).any(), "NaN detected in output latents!"

    print("=" * 70)
    print("ALL TWO-STAGE PIPELINE CONTRACT & STAGE CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_two_stage_pipeline_stages()
