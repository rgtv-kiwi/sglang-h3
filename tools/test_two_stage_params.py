#!/usr/bin/env python3
"""
Unit test for MiniMaxH3SamplingParams two_stage parameter validation and extra field generation.
"""

import sys
from pathlib import Path

# Add sglang/python to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams


def test_params_default():
    params = MiniMaxH3SamplingParams()
    assert params.two_stage_sampling is False
    assert params.two_stage_split_step == 4
    assert params.two_stage_scale_factor == 2.0
    assert params.two_stage_upscaler_path is None
    print("✅ Default parameters test passed.")


def test_params_two_stage_extra():
    params = MiniMaxH3SamplingParams(
        prompt="test prompt",
        task="t2va",
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        two_stage_sampling=True,
        two_stage_split_step=4,
        two_stage_scale_factor=2.0,
        two_stage_upscaler_path="/tmp/upscaler.safetensors",
    )
    extra = params.build_request_extra()
    assert extra.get("two_stage_sampling") is True
    assert extra.get("two_stage_split_step") == 4
    assert extra.get("two_stage_scale_factor") == 2.0
    assert extra.get("two_stage_upscaler_path") == "/tmp/upscaler.safetensors"
    print("✅ Two-stage build_request_extra test passed.")


def test_extra_fields_registered():
    extra_fields = MiniMaxH3SamplingParams.video_request_extra_fields()
    for field_name in [
        "two_stage_sampling",
        "two_stage_split_step",
        "two_stage_scale_factor",
        "two_stage_upscaler_path",
    ]:
        assert field_name in extra_fields, f"Field {field_name} not registered in video_request_extra_fields"
    print("✅ Video request extra fields registration test passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MiniMax-H3 SamplingParams (二采参数) 单元测试")
    print("=" * 60)
    test_params_default()
    test_params_two_stage_extra()
    test_extra_fields_registered()
    print("=" * 60)
    print("🎉 单元测试全部通过 (ALL CHECKS PASSED)！")
    print("=" * 60)
