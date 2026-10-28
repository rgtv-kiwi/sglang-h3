# SPDX-License-Identifier: Apache-2.0
"""Two-stage draft denoise must spatially halve ref2va visual conditions."""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _even_half_latent_dim,
    _halve_ref2va_visual_conditions,
)


def _rows(t: int, h: int, w: int) -> torch.Tensor:
    latent = torch.arange(t * 24 * h * w, dtype=torch.float32).reshape(1, 24, t, h, w)
    return minimax_h3_patchify_video_latent(latent, patch_size=[1, 2, 2])


def test_even_half_matches_draft_canvas_rounding():
    assert _even_half_latent_dim(44, 2.0) == 22
    assert _even_half_latent_dim(80, 2.0) == 40
    assert _even_half_latent_dim(114, 2.0) == 56


def test_halve_ref_image_and_video_tokens():
    image_h, image_w = 8, 8
    video_t, video_h, video_w = 2, 8, 12
    image_rows = _rows(1, image_h, image_w)
    video_rows = _rows(video_t, video_h, video_w)
    visual = torch.cat([image_rows, video_rows], dim=0)
    blocks = [
        {"kind": "image", "latent_h": image_h, "latent_w": image_w},
        {
            "kind": "video",
            "ref_audio_t": 0,
            "latent_t": video_t,
            "latent_h": video_h,
            "latent_w": video_w,
        },
    ]

    new_blocks, new_rows, notes = _halve_ref2va_visual_conditions(
        blocks, visual, scale_factor=2.0
    )
    assert new_blocks[0]["latent_h"] == 4
    assert new_blocks[0]["latent_w"] == 4
    assert new_blocks[1]["latent_h"] == 4
    assert new_blocks[1]["latent_w"] == 6
    assert new_blocks[1]["latent_t"] == video_t
    image_dst = (4 // 2) * (4 // 2)
    video_dst = video_t * (4 // 2) * (6 // 2)
    assert int(new_rows.shape[0]) == image_dst + video_dst
    assert len(notes) == 2

    packed_draft = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        ref_blocks=new_blocks,
    )
    packed_hd = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=3,
        latent_t=2,
        latent_h=8,
        latent_w=12,
        audio_t=3,
        ref_blocks=blocks,
    )
    assert int((~packed_draft["update_mask"]).sum()) == int(new_rows.shape[0])
    assert int((~packed_hd["update_mask"]).sum()) == int(visual.shape[0])
    assert int((~packed_draft["update_mask"]).sum()) < int(
        (~packed_hd["update_mask"]).sum()
    )


def test_audio_only_block_is_unchanged():
    blocks = [{"kind": "audio", "ref_audio_t": 4}]
    new_blocks, new_rows, notes = _halve_ref2va_visual_conditions(
        blocks, None, scale_factor=2.0
    )
    assert new_blocks == blocks
    assert new_rows is None
    assert notes == []
