#!/usr/bin/env python3
"""
Unit test for MiniMaxH3LatentPreparationStage with two-stage support.
Verifies that:
1. Normal mode (two_stage_sampling=False): latent dimensions match requested target.
2. Two-stage mode (two_stage_sampling=True): initial latent dimensions are downscaled by scale_factor (e.g. 2.0x),
   and target dimensions are recorded for transition upscale.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs, set_global_server_args

# Initialize dummy server args for pipeline stage initialization
set_global_server_args(ServerArgs(model_path="/tmp"))


def make_mock_batch(two_stage: bool = False, scale_factor: float = 2.0, split_step: int = 4):
    batch = MagicMock()
    batch.extra = {}
    
    # 768P 16:9 canvas: 1280x720 or 1360x768 -> say 1280x768 -> latent: H=48, W=80
    plan = MagicMock()
    plan.seed = 42
    plan.shape = {
        "geometry": "resolved_v2",
        "height": 768,
        "width": 1280,
        "video_latent_t": 17,
        "audio_latent_t": 8,
    }
    batch.extra["minimax_h3_resolved_plan"] = plan
    
    params = MiniMaxH3SamplingParams(
        prompt="test prompt",
        task="t2va",
        two_stage_sampling=two_stage,
        two_stage_split_step=split_step,
        two_stage_scale_factor=scale_factor,
    )
    batch.sampling_params = params
    return batch


def test_normal_preparation():
    stage = MiniMaxH3LatentPreparationStage()
    batch = make_mock_batch(two_stage=False)
    stage._prepare_denoise_state_from_plan(batch, batch.extra["minimax_h3_resolved_plan"])
    
    state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
    assert state["two_stage_enabled"] is False
    assert state["latent_h"] == 768 // 16  # 48
    assert state["latent_w"] == 1280 // 16 # 80
    assert state["latent_t"] == 17
    print("✅ Normal mode (two_stage=False) preparation test passed!")


def test_two_stage_preparation():
    stage = MiniMaxH3LatentPreparationStage()
    batch = make_mock_batch(two_stage=True, scale_factor=2.0, split_step=4)
    stage._prepare_denoise_state_from_plan(batch, batch.extra["minimax_h3_resolved_plan"])
    
    state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
    assert state["two_stage_enabled"] is True
    assert state["two_stage_split_step"] == 4
    assert state["two_stage_scale_factor"] == 2.0
    
    # Target latent: 48, 80
    assert state["target_latent_h"] == 48
    assert state["target_latent_w"] == 80
    
    # Initial Draft latent: 48 / 2 = 24, 80 / 2 = 40
    assert state["latent_h"] == 24, f"Expected 24, got {state['latent_h']}"
    assert state["latent_w"] == 40, f"Expected 40, got {state['latent_w']}"
    assert state["latent_t"] == 17
    
    # Video noise rows count check: latent_t * (H//2) * (W//2) = 17 * (24//2) * (40//2) = 17 * 12 * 20 = 4080
    expected_rows = 17 * (24 // 2) * (40 // 2)
    assert state["initial_video_rows"].shape[0] == expected_rows, (
        f"Row count mismatch: expected {expected_rows}, got {state['initial_video_rows'].shape[0]}"
    )
    print("✅ Two-stage mode (two_stage=True) downscaled latent preparation test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 MiniMax-H3 LatentPreparationStage (二采潜空间准备) 单元测试")
    print("=" * 70)
    test_normal_preparation()
    test_two_stage_preparation()
    print("=" * 70)
    print("🎉 单元测试全部通过 (ALL CHECKS PASSED)！")
    print("=" * 70)
