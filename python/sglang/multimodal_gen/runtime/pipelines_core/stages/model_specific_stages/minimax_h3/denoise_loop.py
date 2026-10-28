# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 cfg-distilled full denoise loop.

Per step, the positive presentation is forwarded exactly once. Video and audio
target rows chain through the Euler-eta0 update while visual and audio condition
rows stay pinned to their noised step-0 anchors.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)

MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
# ref2va audio reference anchor timestep
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
# Packed row widths: video rows are [1,2,2]-patchified 24-channel latents
# (24 * 1 * 2 * 2 = 96); audio rows carry the 32-dim audio latent.
MINIMAX_H3_VIDEO_ROW_WIDTH = 96
MINIMAX_H3_AUDIO_ROW_WIDTH = 32
_MINIMAX_H3_SUBBLOCK_QUERY_BLOCK_SIZE = 64


def _minimax_h3_subblock_video_query_indices(
    packed: dict[str, torch.Tensor],
    text_video_token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return packed video rows plus any Qwen rows marked as video."""
    latent_video_pos = packed["video_pos"].view(-1).to(dtype=torch.long)
    if text_video_token_mask is None:
        return latent_video_pos
    text_pos = packed["text_pos"].view(-1).to(dtype=torch.long)
    text_video_token_mask = text_video_token_mask.view(-1).to(
        device=text_pos.device,
        dtype=torch.bool,
    )
    if text_video_token_mask.shape[0] != text_pos.shape[0]:
        raise ValueError(
            "text_video_token_mask must align with packed text rows "
            f"({text_pos.shape[0]}), got {text_video_token_mask.shape[0]}"
        )
    return torch.cat([text_pos[text_video_token_mask], latent_video_pos])


def _minimax_h3_subblock_sparse_query_block_mask(
    video_query_indices: torch.Tensor,
    *,
    used_len: int,
) -> torch.Tensor:
    """Return True for pure-video 64-row Q blocks and False otherwise."""
    if used_len < 0:
        raise ValueError(f"used_len must be non-negative, got {used_len}")
    if video_query_indices.ndim != 1:
        raise ValueError(
            f"video_query_indices must be rank 1, got {list(video_query_indices.shape)}"
        )
    video_query_indices = video_query_indices.to(dtype=torch.long)
    if video_query_indices.numel():
        first = int(video_query_indices.min())
        last = int(video_query_indices.max())
        if first < 0 or last >= used_len:
            raise ValueError(
                "video_query_indices must be first-segment-relative and in "
                f"[0, {used_len}), got min={first}, max={last}"
            )
        if torch.unique(video_query_indices).numel() != video_query_indices.numel():
            raise ValueError("video_query_indices must not contain duplicates")
    block_size = _MINIMAX_H3_SUBBLOCK_QUERY_BLOCK_SIZE
    num_query_blocks = -(-used_len // block_size)
    video_rows_per_block = torch.bincount(
        torch.div(video_query_indices, block_size, rounding_mode="floor"),
        minlength=num_query_blocks,
    )
    block_ids = torch.arange(
        num_query_blocks, device=video_query_indices.device, dtype=torch.long
    )
    real_rows_per_block = (used_len - block_ids * block_size).clamp(
        min=0, max=block_size
    )
    # BSA is block-granular. Only blocks whose real rows are all video may use
    # the sparse budget; mixed boundary blocks stay dense so their non-video
    # rows retain exact attention in the single heterogeneous BSA call.
    return video_rows_per_block == real_rows_per_block


@torch.inference_mode()
def _minimax_h3_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_t: torch.Tensor,
    sigma_curr: float,
    sigma_ratio: torch.Tensor,
    one_minus_sigma_ratio: torch.Tensor,
    denoised_scratch: torch.Tensor,
) -> None:
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    if sigma_curr == 0.0:
        return
    torch.mul(one_minus_sigma_ratio, denoised_scratch, out=velocity)
    torch.mul(sigma_ratio, state, out=state)
    torch.add(state, velocity, out=state)


def _build_local_embedding_layout(
    *,
    seq_len: int,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if seq_len % world_size:
        raise ValueError(
            f"packed seq_len {seq_len} not divisible by the combined "
            f"sequence-parallel world size {world_size}"
        )
    local_seq_len = seq_len // world_size
    row_start = rank * local_seq_len
    row_stop = row_start + local_seq_len

    def local_ids(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = torch.nonzero(
            (pos >= row_start) & (pos < row_stop),
            as_tuple=False,
        ).view(-1)
        return source_ids.to(device), pos.index_select(0, source_ids).to(device)

    text_source_start = min(row_start, int(text_pos.shape[0]))
    text_source_stop = min(row_stop, int(text_pos.shape[0]))
    _, img_global_ids = local_ids(img_pos)
    _, audio_global_ids = local_ids(audio_pos)
    return {
        "text_source_start": text_source_start,
        "text_source_stop": text_source_stop,
        "img_global_ids": img_global_ids,
        "img_row_ids": img_global_ids - row_start,
        "audio_global_ids": audio_global_ids,
        "audio_row_ids": audio_global_ids - row_start,
    }


class MiniMaxH3DenoiseBranch:
    """Static per-branch state: packed layout + fixed forward kwargs.

    `packed` is a minimax_h3_packed_sequence(...) result (or equivalent layout
    dict); `text_embeddings` is the branch's [text_len, 5120] hidden states;
    `token_tags` must already carry any fl2va vision-span overrides.
    """

    def __init__(
        self,
        *,
        packed: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        device: torch.device,
        video_query_indices: torch.Tensor | None = None,
    ) -> None:
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(torch.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        self.update_mask = packed["update_mask"].view(-1).to(torch.bool)
        # ref2va: audio_pos may include reference-audio anchor rows
        # (audio_update_mask False); absent means all rows are targets.
        if "audio_update_mask" in packed:
            self.audio_update_mask = packed["audio_update_mask"].view(-1).to(torch.bool)
        else:
            self.audio_update_mask = torch.ones(
                self.audio_pos.shape[0], dtype=torch.bool
            )
        # Packed H3 layouts place reference rows before the generated suffix
        # for both modalities. Keep the suffix boundary once so the denoise
        # hot path can update contiguous views instead of gathering and
        # scattering the same static index tensors on every step.
        self.video_target_start = int((~self.update_mask).sum())
        self.audio_target_start = int((~self.audio_update_mask).sum())
        self.video_target_slice = slice(self.video_target_start, None)
        self.audio_target_slice = slice(self.audio_target_start, None)
        text_pos = packed["text_pos"].view(-1).to(torch.long)
        text_len = int(text_pos.shape[0])
        if list(text_embeddings.shape)[0] != text_len:
            raise ValueError(
                f"text_embeddings rows {list(text_embeddings.shape)} != "
                f"packed text_len {text_len}"
            )
        if int(token_tags.view(-1).shape[0]) != seq_len:
            raise ValueError(
                f"token_tags length {int(token_tags.view(-1).shape[0])} != "
                f"seq_len {seq_len}"
            )
        cu = packed["cu_seqlens"].to(torch.int32)
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        # Resolve the remaining step-static packed-sequence and anchor row
        # sets once, keeping nonzero-driven work out of the hot loop.
        self.img_cond_seq_idx = self.img_pos_dev[~self.update_mask_dev]
        self.img_target_seq_idx = self.img_pos_dev[self.update_mask_dev]
        self.audio_target_seq_idx = self.audio_pos_dev[self.audio_update_mask_dev]
        self.audio_ref_seq_idx = self.audio_pos_dev[~self.audio_update_mask_dev]
        self.cond_row_idx = torch.nonzero(~self.update_mask_dev).view(-1)
        self.audio_ref_row_idx = torch.nonzero(~self.audio_update_mask_dev).view(-1)
        # rows that keep the video timestep each step: text, padding, and
        # video target rows — everything the three overwrite sets do not cover
        self.n_video_timestep_rows = (
            seq_len
            - int(self.img_cond_seq_idx.numel())
            - int(self.audio_target_seq_idx.numel())
            - int(self.audio_ref_seq_idx.numel())
        )
        # persistent packed-row buffers; every img/audio position is fully
        # rewritten by index_copy_ on the first forward_kwargs() call, then
        # only the target-row subset each step after (condition/reference
        # rows never change post-priming -- see forward_kwargs).
        self._x_buffer_primed = False
        self.x_buffer = torch.zeros(
            1, seq_len, MINIMAX_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device
        )
        self.audio_x_buffer = torch.zeros(
            1, seq_len, MINIMAX_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device
        )
        text_pos_dev = text_pos.to(device)
        ulysses_world_size, ulysses_rank = get_ulysses_ctx()
        ring_world_size, ring_rank = get_ring_ctx()
        # Combined SP-local rank/world size: the group coordinator lays out
        # ring as the outer (slower-varying) dimension and Ulysses as the
        # inner one (see set_seq_parallel_pg_by_sp_groups), so this matches
        # minimax_h3.py's row_start = ring_rank*ring_chunk_len +
        # ulysses_rank*local_seq_len exactly -- `_build_local_embedding_layout`
        # below needs the same combined rank, not just the Ulysses component.
        sp_world_size = ulysses_world_size * ring_world_size
        sp_rank = ring_rank * ulysses_world_size + ulysses_rank
        token_tags_host = token_tags.view(-1).to(dtype=torch.long)
        subblock_sparse_query_block_mask = (
            _minimax_h3_subblock_sparse_query_block_mask(
                video_query_indices.view(-1).to(dtype=torch.long),
                used_len=int(cu[1]),
            ).to(device)
            if video_query_indices is not None
            else None
        )
        local_seq_len = seq_len // sp_world_size
        local_row_start = sp_rank * local_seq_len
        local_row_stop = local_row_start + local_seq_len
        self.local_row_slice = slice(local_row_start, local_row_stop)
        self.block_token_tags = (
            token_tags_host[local_row_start:local_row_stop].clamp(min=0).to(device)
        )
        # Attention sees the whole packed sequence on every rank -- Ulysses
        # trades sequence for heads inside it, so the row shard is restored to
        # full length before the kernel runs. Backends that need per-row tags
        # there (sparge_attn keeps text/audio dense) therefore need the
        # full-length copy, not this rank's slice. Only carried when the two
        # actually differ; without sequence parallelism block_token_tags is
        # already the whole sequence.
        self.packed_token_tags = (
            token_tags_host.clamp(min=0).to(device) if sp_world_size > 1 else None
        )
        self.static_kwargs: dict[str, Any] = {
            # Cast the fp64 position grid on the host. MiniMaxH3Rope casts it to
            # fp32 as its first op anyway, so the values are identical; doing the
            # cast on CPU also avoids platforms that cannot execute fp64 on
            # device (e.g. Iluvatar CoreX returns zeros for device fp64).
            "img_position_ids": packed["img_position_ids"][None]
            .to(torch.float32)
            .to(device),
            "update_mask": self.update_mask_dev,
            "block_token_tags": self.block_token_tags,
            "skip_mask_out_condition": True,
            "prompt_embeds": text_embeddings.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": text_pos_dev},
            "img_pos_for_infer_output_info": {"position_ids": self.img_target_seq_idx},
            "local_embedding_layout": _build_local_embedding_layout(
                seq_len=seq_len,
                text_pos=text_pos,
                img_pos=self.img_pos,
                audio_pos=self.audio_pos,
                world_size=sp_world_size,
                rank=sp_rank,
                device=device,
            ),
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "cu_seqlens_q_host": tuple(int(value) for value in cu.tolist()),
                "max_seqlen_q": int(cu[1]),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor(
                    [0, text_len, text_len], dtype=torch.int32, device=device
                ),
                "cu_seqlens_q_host": (0, text_len, text_len),
                "max_seqlen_q": text_len,
            },
        }
        if self.packed_token_tags is not None:
            self.static_kwargs["token_tags"] = self.packed_token_tags
        if subblock_sparse_query_block_mask is not None:
            self.static_kwargs["subblock_sparse_query_block_mask"] = (
                subblock_sparse_query_block_mask
            )

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        step_timesteps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        adaln_slot: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        x = self.x_buffer
        audio_x = self.audio_x_buffer
        if not self._x_buffer_primed:
            # First step: condition/reference rows have just been pinned into
            # video_rows/audio_rows (see minimax_h3_denoise_loop) and never
            # change again, so this is the only step that needs the full
            # img/audio extent written into the persistent buffers.
            x[0].index_copy_(0, self.img_pos_dev, video_rows)
            audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
            self._x_buffer_primed = True
        else:
            # Later steps: only the target-row subset changed since the
            # buffers were primed; rewriting condition/reference rows again
            # would just copy the same bytes already sitting there.
            x[0].index_copy_(
                0, self.img_target_seq_idx, video_rows[self.video_target_slice]
            )
            audio_x[0].index_copy_(
                0, self.audio_target_seq_idx, audio_rows[self.audio_target_slice]
            )
        unique_timesteps, inverse_indices, block_combined_indices = step_timesteps
        kwargs = {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
            "block_combined_indices": block_combined_indices,
        }
        if adaln_slot is not None:
            # Device scalar, never a Python int: an int would key one breakable
            # CUDA graph per slot value and go stale when LRU reuses the slot.
            kwargs["adaln_cache_slot"] = adaln_slot
        return kwargs

    def _expand_step_timesteps(
        self,
        *,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor],
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build step-local timestep and AdaLN index tensors.

        Packed-sequence timestep semantics: non-media rows (text and padding)
        inherit the current video timestep, condition rows pin their noise-aug
        timesteps. The row->timestep layout is step-static, so instead of
        materializing the full timestep tensor and paying a device-syncing
        torch.unique per step, the at-most-four candidate values are deduped
        in fp32 on the host — torch.unique on the candidate tensor keeps exact
        fp32 collision semantics — and inverse indices are index_fill'ed from
        the static position sets.
        """
        candidates: list[float] = []
        fill_groups: list[tuple[torch.Tensor, int]] = []
        base_slot = -1
        if self.n_video_timestep_rows > 0:
            base_slot = len(candidates)
            candidates.append(float(t_video))
        for seq_idx, value in (
            (self.img_cond_seq_idx, imgvid_cond_timestep),
            (self.audio_target_seq_idx, t_audio),
            (self.audio_ref_seq_idx, audio_ref_cond_timestep),
        ):
            if seq_idx.numel() > 0:
                fill_groups.append((seq_idx, len(candidates)))
                candidates.append(float(value))
        unique_cpu, slot_to_unique = torch.unique(
            torch.tensor(candidates, dtype=torch.float32),
            sorted=True,
            return_inverse=True,
        )
        device = self.img_pos_dev.device
        base_index = int(slot_to_unique[base_slot]) if base_slot >= 0 else 0
        pattern = tuple(slot_to_unique.tolist())
        inverse_indices = inverse_indices_by_pattern.get(pattern)
        if inverse_indices is None:
            inverse_indices = torch.full(
                (self.seq_len,), base_index, dtype=torch.long, device=device
            )
            for seq_idx, slot in fill_groups:
                inverse_indices.index_fill_(0, seq_idx, int(slot_to_unique[slot]))
            inverse_indices_by_pattern[pattern] = inverse_indices
        block_combined = block_combined_by_pattern.get(pattern)
        if block_combined is None:
            block_combined = torch.add(
                self.block_token_tags,
                inverse_indices[self.local_row_slice],
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            block_combined_by_pattern[pattern] = block_combined
        return unique_cpu.to(device), inverse_indices, block_combined

    def prepare_timestep_plan(
        self,
        *,
        video_timesteps: list[float],
        audio_timesteps: list[float],
        imgvid_cond_noise_aug: float,
        audio_ref_cond_noise_aug: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Stage every step's packed timestep state before denoising."""
        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("video/audio timestep plans must have equal length")
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        return [
            self._expand_step_timesteps(
                t_video=t_video,
                t_audio=t_audio,
                imgvid_cond_timestep=max(t_video, imgvid_cond_noise_aug),
                audio_ref_cond_timestep=max(t_audio, audio_ref_cond_noise_aug),
                inverse_indices_by_pattern=inverse_indices_by_pattern,
                block_combined_by_pattern=block_combined_by_pattern,
            )
            for t_video, t_audio in zip(video_timesteps, audio_timesteps)
        ]


def minimax_h3_denoise_loop(
    *,
    model: Any,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    stage2_keyframe_cond_rows: torch.Tensor | None = None,
    stage2_keyframe_latent_hw: tuple[int, int] | None = None,
    raw_keyframe_rows: torch.Tensor | None = None,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None = None,
    keyframe_frame_count: int | None = None,
    stage2_layout_cond_rows: torch.Tensor | None = None,
    audio_ref_rows: torch.Tensor | None = None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_for_inference: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    attn_metadata: AttentionMetadata | None = None,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
    two_stage_manager: Any | None = None,
    rebuild_layout_fn: (
        Callable[[int, int], tuple[MiniMaxH3DenoiseBranch, AttentionMetadata | None]]
        | None
    ) = None,
    current_latent_dims: tuple[int, int, int] | None = None,
    target_latent_hw: tuple[int, int] | None = None,
    output_state: dict[str, Any] | None = None,
    seed: int = 42,
    model_forward: (
        Callable[[Any, dict[str, Any], int], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full denoise loop; returns final (video_rows, audio_rows).

    ``initial_video_rows`` covers all image rows of the positive layout. For a
    conditional task, pass ``keyframe_cond_rows`` and/or ``audio_ref_rows`` to
    pin those rows across every step. The model's raw positive velocity is the
    update signal; MiniMax H3 only supports cfg-distilled checkpoints.
    ``model_forward`` is the native-stage hook for residency/BCG runners and
    receives the zero-based loop step; the default keeps this helper
    independently testable with a plain callable.
    """
    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    n_cond = positive.video_target_start
    if keyframe_cond_rows is None:
        if n_cond != 0:
            raise ValueError(
                f"layout has {n_cond} cond rows but keyframe_cond_rows is None"
            )
    else:
        if int(keyframe_cond_rows.shape[0]) != n_cond:
            raise ValueError(
                f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} != "
                f"layout cond rows {n_cond}"
            )
    video_rows = initial_video_rows.to(device=device, dtype=torch.float32, copy=True)
    audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32, copy=True)
    if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
        raise ValueError(
            f"initial video rows {int(video_rows.shape[0])} != positive layout "
            f"rows {int(positive.img_pos.shape[0])}"
        )
    if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
        raise ValueError(
            f"initial audio rows {int(audio_rows.shape[0])} != positive layout "
            f"rows {int(positive.audio_pos.shape[0])}"
        )
    n_audio_ref = positive.audio_target_start
    if audio_ref_rows is None:
        if n_audio_ref != 0:
            raise ValueError(
                f"layout has {n_audio_ref} audio ref rows but audio_ref_rows is None"
            )
        audio_anchor = None
    else:
        if int(audio_ref_rows.shape[0]) != n_audio_ref:
            raise ValueError(
                f"audio_ref_rows {int(audio_ref_rows.shape[0])} != layout "
                f"audio ref rows {n_audio_ref}"
            )
        audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
    cond_anchor = (
        keyframe_cond_rows.to(device=device, dtype=torch.float32)
        if keyframe_cond_rows is not None
        else None
    )
    if cond_anchor is not None:
        video_rows.index_copy_(0, positive.cond_row_idx, cond_anchor)
    if audio_anchor is not None:
        audio_rows.index_copy_(0, positive.audio_ref_row_idx, audio_anchor)

    num_steps = len(sigmas_video) - 1
    video_target_slice = positive.video_target_slice
    audio_target_slice = positive.audio_target_slice
    video_timesteps = [1.0 - sigma for sigma in sigmas_video[:-1]]
    audio_timesteps = [1.0 - sigma for sigma in sigmas_audio[:-1]]
    # One H2D copy per schedule, preserving the previous Python-float
    # subtraction followed by fp32 conversion.
    video_step_t = torch.tensor(video_timesteps, dtype=torch.float32, device=device)
    audio_step_t = torch.tensor(audio_timesteps, dtype=torch.float32, device=device)
    timestep_plan = positive.prepare_timestep_plan(
        video_timesteps=video_timesteps,
        audio_timesteps=audio_timesteps,
        imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
        audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
    )
    # Every step's timesteps are settled by now. Rebuilding AdaLN reads all
    # 24.2 GiB of adaln_proj whatever is missing, so fill the whole request in
    # one pass here instead of topping up step by step inside the loop.
    adaln_plan_slots = model.prepare_adaln_plans([entry[0] for entry in timestep_plan])

    # match the scheduler's device-fp32 math once, then reuse one denoised
    # scratch per modality instead of allocating intermediates every step
    video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
    audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
    video_sigma_ratios = video_sigmas[1:] / video_sigmas[:-1]
    audio_sigma_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
    video_sigma_t = 1.0 - video_step_t
    audio_sigma_t = 1.0 - audio_step_t
    video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
    audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios
    video_denoised_scratch = torch.empty_like(video_rows[video_target_slice])
    audio_denoised_scratch = torch.empty_like(audio_rows[audio_target_slice])
    from tqdm import tqdm

    is_two_stage = (
        two_stage_manager is not None
        and getattr(two_stage_manager, "enabled", False)
        and rebuild_layout_fn is not None
        and current_latent_dims is not None
    )

    split_step = getattr(two_stage_manager, "split_step", 4) if is_two_stage else num_steps

    def _execute_step(step: int) -> None:
        nonlocal video_rows, audio_rows, video_denoised_scratch, audio_denoised_scratch
        nonlocal positive, attn_metadata, timestep_plan, video_target_slice, audio_target_slice
        nonlocal sigmas_video, sigmas_audio, adaln_plan_slots
        nonlocal video_sigma_t, audio_sigma_t, video_sigma_ratios, audio_sigma_ratios
        nonlocal video_one_minus_sigma_ratios, audio_one_minus_sigma_ratios

        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            s_v = sigmas_video[step]
            s_a = sigmas_audio[step]

            fk = positive.forward_kwargs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                step_timesteps=timestep_plan[step],
                adaln_slot=(
                    None if adaln_plan_slots is None else adaln_plan_slots[step]
                ),
            )
            if attn_metadata is not None:
                attn_metadata.current_timestep = step
            if model_forward is None and attn_metadata is not None:
                forward_cm: AbstractContextManager = set_forward_context(
                    current_timestep=step,
                    attn_metadata=attn_metadata,
                )
            else:
                forward_cm = nullcontext()
            with forward_cm, torch.inference_mode():
                if model_forward is None:
                    v_video, v_audio = model(**fk)
                else:
                    v_video, v_audio = model_forward(model, fk, step)
                # The model outputs are inference tensors. Keep their disposable
                # fp32 velocity updates in the same context so ``out=velocity``
                # can reuse the output storage without an extra clone.
                mv_video_t = v_video.float()
                mv_audio_t = v_audio[audio_target_slice].float()

                video_target = video_rows[video_target_slice]
                _minimax_h3_update_target_rows_(
                    video_target,
                    mv_video_t,
                    sigma_t=video_sigma_t[step],
                    sigma_curr=s_v,
                    sigma_ratio=video_sigma_ratios[step],
                    one_minus_sigma_ratio=video_one_minus_sigma_ratios[step],
                    denoised_scratch=video_denoised_scratch,
                )

                audio_target = audio_rows[audio_target_slice]
                _minimax_h3_update_target_rows_(
                    audio_target,
                    mv_audio_t,
                    sigma_t=audio_sigma_t[step],
                    sigma_curr=s_a,
                    sigma_ratio=audio_sigma_ratios[step],
                    one_minus_sigma_ratio=audio_one_minus_sigma_ratios[step],
                    denoised_scratch=audio_denoised_scratch,
                )
            if on_step is not None:
                on_step(step, video_rows, audio_rows)

    if not is_two_stage:
        for step in range(num_steps):
            _execute_step(step)
    else:
        # ======================================================================
        # 两阶段采样调度逻辑 (Two-Stage Sampling)
        # 支持两种模式：
        # 1. full_x0 (默认): 一采完整去噪到 sigma=0.0，获得真值纯净 x0 送入放大器，加噪后再进入二采高清精修；
        # 2. 原版预测模式: 一采中途单步外推 x0 = xt - sigma*v 送入放大器。
        # ======================================================================
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_world_rank,
            world_group_is_initialized,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_time_shift_sigmas,
        )

        is_master = not world_group_is_initialized() or get_world_rank() == 0

        def _log_master(*args, **kwargs):
            if is_master:
                print(*args, **kwargs)

        full_x0_mode = (
            os.environ.get("MINIMAX_H3_TWO_STAGE_FULL_X0", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )

        if not full_x0_mode:
            # ----------------- 回退：原版中途预测截断模式 -----------------
            stage1_steps = min(split_step, num_steps)
            _log_master(
                f"\n{'='*75}\n"
                f">>> [TwoStage-Legacy] 启动 Stage 1/2: 一采 Draft 低分辨率采样 (共 {stage1_steps} 步，中途外推预测)\n"
                f"{'='*75}",
                flush=True,
            )
            with tqdm(
                total=stage1_steps,
                desc="[Stage 1/2 一采 Draft (0.2MP)]",
                dynamic_ncols=True,
                disable=not is_master,
            ) as pbar1:
                for step in range(0, stage1_steps):
                    _execute_step(step)
                    if is_master:
                        pbar1.update(1)

            if stage1_steps < num_steps:
                latent_t, cur_h, cur_w = current_latent_dims
                sigma_next = float(sigmas_video[stage1_steps])
                clean_video_draft = video_denoised_scratch

                new_video_target, new_h, new_w = two_stage_manager.upscale_and_repack(
                    clean_video_draft,
                    latent_t=latent_t,
                    latent_h=cur_h,
                    latent_w=cur_w,
                    target_latent_hw=target_latent_hw,
                )
                current_latent_dims = (latent_t, new_h, new_w)
                new_positive, new_attn_metadata = rebuild_layout_fn(new_h, new_w)
                positive = new_positive
                attn_metadata = new_attn_metadata
                video_target_slice = positive.video_target_slice

                noise_seed = (int(seed) + stage1_steps * 10007) & 0x7FFFFFFF
                gen_noise = torch.Generator(device="cpu").manual_seed(noise_seed)
                target_noise = torch.randn(
                    new_video_target.shape, generator=gen_noise, dtype=torch.float32
                ).to(device=device, dtype=new_video_target.dtype)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.broadcast(target_noise, src=0)

                custom_sigma_env = os.environ.get("MINIMAX_H3_TWO_STAGE_NOISE_SIGMA", "").strip()
                actual_sigma = float(custom_sigma_env) if custom_sigma_env else sigma_next
                new_video_target = (1.0 - actual_sigma) * new_video_target + actual_sigma * target_noise

                transition_cond = cond_anchor
                if stage2_layout_cond_rows is not None:
                    transition_cond = stage2_layout_cond_rows.to(device=device, dtype=torch.float32)
                new_video_rows = torch.zeros(
                    int(positive.img_pos.shape[0]),
                    new_video_target.shape[1],
                    dtype=torch.float32,
                    device=device,
                )
                if transition_cond is not None:
                    new_video_rows.index_copy_(0, positive.cond_row_idx, transition_cond)
                new_video_rows[video_target_slice] = new_video_target
                video_rows = new_video_rows
                video_denoised_scratch = torch.empty_like(video_rows[video_target_slice])

                timestep_plan = positive.prepare_timestep_plan(
                    video_timesteps=video_timesteps,
                    audio_timesteps=audio_timesteps,
                    imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
                    audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
                )
                adaln_plan_slots = model.prepare_adaln_plans([entry[0] for entry in timestep_plan])

                stage2_steps = num_steps - stage1_steps
                with tqdm(
                    total=stage2_steps,
                    desc=f"[Stage 2/2 二采 高清精修 ({new_h*16}P)]",
                    dynamic_ncols=True,
                    disable=not is_master,
                ) as pbar2:
                    for step in range(stage1_steps, num_steps):
                        _execute_step(step)
                        if is_master:
                            pbar2.update(1)
        else:
            # ==================================================================
            # 【全新 Cascaded 方案】一采完整降噪到 0.0 -> 上采样 -> 加噪 -> 二采高清精修
            # ==================================================================
            stage1_steps = int(os.environ.get(
                "MINIMAX_H3_TWO_STAGE_STAGE1_STEPS", str(split_step)
            ))
            stage2_steps = int(os.environ.get(
                "MINIMAX_H3_TWO_STAGE_STAGE2_STEPS", str(max(1, num_steps - split_step))
            ))

            v_shift = 12.0
            a_shift = 3.0

            # 1. 为 Stage 1 构造独立且完整的 1.0 -> 0.0 去噪调度 (严格 stage1_steps 步去噪 = stage1_steps + 1 个节点)
            base_s1 = torch.linspace(1.0, 0.0, stage1_steps + 1, device="cpu", dtype=torch.float32)
            shifted_v = float(v_shift) * base_s1 / (1 + (float(v_shift) - 1) * base_s1)
            s1_sigmas_v = [float(v) for v in shifted_v.tolist()]
            s1_sigmas_v[-1] = 0.0

            shifted_a = float(a_shift) * base_s1 / (1 + (float(a_shift) - 1) * base_s1)
            s1_sigmas_a = [float(a) for a in shifted_a.tolist()]
            s1_sigmas_a[-1] = 0.0

            stage1_actual_steps = len(s1_sigmas_v) - 1

            # 更新 Stage 1 专属的时间步与 AdaLN Plans
            sigmas_video = s1_sigmas_v
            sigmas_audio = s1_sigmas_a
            video_timesteps = [1.0 - s for s in sigmas_video[:-1]]
            audio_timesteps = [1.0 - s for s in sigmas_audio[:-1]]
            video_step_t = torch.tensor(video_timesteps, dtype=torch.float32, device=device)
            audio_step_t = torch.tensor(audio_timesteps, dtype=torch.float32, device=device)
            timestep_plan = positive.prepare_timestep_plan(
                video_timesteps=video_timesteps,
                audio_timesteps=audio_timesteps,
                imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
                audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
            )
            adaln_plan_slots = model.prepare_adaln_plans([entry[0] for entry in timestep_plan])

            video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
            audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
            video_sigma_ratios = video_sigmas[1:] / video_sigmas[:-1]
            audio_sigma_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
            video_sigma_t = 1.0 - video_step_t
            audio_sigma_t = 1.0 - audio_step_t
            video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
            audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios

            _log_master(
                f"\n{'='*75}\n"
                f">>> [TwoStage-FullX0] 启动 Stage 1/2: 一采 Draft 低分辨率采样 (共 {stage1_actual_steps} 步，完整去噪至 sigma=0.0)\n"
                f"    [+] Schedule: video_sigmas[0]={s1_sigmas_v[0]:.4f} -> video_sigmas[-1]={s1_sigmas_v[-1]:.4f}\n"
                f"{'='*75}",
                flush=True,
            )
            with tqdm(
                total=stage1_actual_steps,
                desc="[Stage 1/2 一采 Draft 完整去噪至 0]",
                dynamic_ncols=True,
                disable=not is_master,
            ) as pbar1:
                for step in range(0, stage1_actual_steps):
                    _execute_step(step)
                    if is_master:
                        pbar1.update(1)

            # ==================================================================
            # 【转折点 Transition Point】真值 X0 导出、3D 潜空间上采样、加噪注入
            # ==================================================================
            latent_t, cur_h, cur_w = current_latent_dims

            # 经过完整去噪，video_rows[video_target_slice] 此时为自然收敛的纯净真值 x0
            clean_video_draft = video_rows[video_target_slice].clone()

            if output_state is not None and os.environ.get(
                "MINIMAX_H3_DUMP_STAGE1_X0", ""
            ).strip() not in ("", "0", "false", "False"):
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
                    minimax_h3_unpatchify_video_tokens,
                )

                output_state["stage1_x0_latents"] = minimax_h3_unpatchify_video_tokens(
                    clean_video_draft.detach(),
                    latent_shape=[latent_t, cur_h // 2, cur_w // 2, 24],
                    patch_size=[1, 2, 2],
                ).contiguous()
                output_state["stage1_x0_latent_hw"] = (int(cur_h), int(cur_w))
                _log_master(
                    f"    [*] [FullX0] 导出 Stage 1 完整去噪纯净潜空间 x0: {tuple(output_state['stage1_x0_latents'].shape)} "
                    f"(pixel ~{cur_w * 16}x{cur_h * 16})\n",
                    flush=True,
                )

            _log_master(
                f"\n{'='*75}\n"
                f">>> [TwoStage-FullX0] Stage 1 完整去噪完毕，开始放大与二采准备:\n"
                f"    [1/4] 将 100% 完整去噪的干净潜空间送入 3D Latent Upscaler 放大 (2.0x)...\n",
                flush=True,
            )

            # 3D 卷积放大
            new_video_target, new_h, new_w = two_stage_manager.upscale_and_repack(
                clean_video_draft,
                latent_t=latent_t,
                latent_h=cur_h,
                latent_w=cur_w,
                target_latent_hw=target_latent_hw,
            )
            current_latent_dims = (latent_t, new_h, new_w)

            _log_master(
                f"    [2/4] 潜空间放大完成: ({cur_h}x{cur_w} -> {new_h}x{new_w})\n"
                f"    [3/4] 动态重构高清序列 Packed 布局与 3D RoPE Cache...\n",
                flush=True,
            )

            # 重建高清布局
            new_positive, new_attn_metadata = rebuild_layout_fn(new_h, new_w)
            positive = new_positive
            attn_metadata = new_attn_metadata
            video_target_slice = positive.video_target_slice

            # 首尾帧真值锚定 (方案 A: 默认不再硬锁，防止破坏 3D 上采样后的时序平滑度)
            lock_keyframes = (
                os.environ.get("MINIMAX_H3_LOCK_KEYFRAMES", "0").strip().lower()
                not in ("0", "false", "no", "off")
            )
            if lock_keyframes and raw_keyframe_rows is not None and keyframe_frame_indices:
                frame_rows = (new_h // 2) * (new_w // 2)
                for blk_idx, frame_idx in enumerate(keyframe_frame_indices):
                    src_slice = slice(blk_idx * frame_rows, (blk_idx + 1) * frame_rows)
                    clean_cond_frame = raw_keyframe_rows[src_slice].to(
                        device=device, dtype=new_video_target.dtype
                    )
                    if frame_idx == 0:
                        new_video_target[0:frame_rows] = clean_cond_frame
                        _log_master(
                            "    [*] [首帧真值注入] 视频第 0 帧在加噪前已锚定为 100% 原始高清原图潜空间！\n",
                            flush=True,
                        )
                    elif frame_idx == -1 or (
                        keyframe_frame_count and frame_idx == keyframe_frame_count - 1
                    ):
                        dst_slice = slice((latent_t - 1) * frame_rows, latent_t * frame_rows)
                        new_video_target[dst_slice] = clean_cond_frame
                        _log_master(
                            "    [*] [尾帧真值注入] 视频末帧在加噪前已锚定为 100% 原始高清尾图潜空间！\n",
                            flush=True,
                        )
            elif not lock_keyframes:
                _log_master(
                    "    [*] [方案A-时序平滑] 保持 Stage 1 完整去噪与 3D 放大后的连续潜空间，不执行前置首尾帧硬替换。\n",
                    flush=True,
                )

            # Dump 放大后的干净 X0
            if output_state is not None and os.environ.get(
                "MINIMAX_H3_DUMP_STAGE1_X0", ""
            ).strip() not in ("", "0", "false", "False"):
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
                    minimax_h3_unpatchify_video_tokens,
                )

                output_state["stage1_upscaled_x0_latents"] = minimax_h3_unpatchify_video_tokens(
                    new_video_target.detach(),
                    latent_shape=[latent_t, new_h // 2, new_w // 2, 24],
                    patch_size=[1, 2, 2],
                ).contiguous()
                output_state["stage1_upscaled_x0_latent_hw"] = (int(new_h), int(new_w))
                _log_master(
                    f"    [*] [FullX0] 导出 Stage 1 放大后纯净 x0: {tuple(output_state['stage1_upscaled_x0_latents'].shape)}\n",
                    flush=True,
                )

            # ==================================================================
            # 【Stage 2 Schedule 解析】X1 优化版：平滑 S 型非线性降噪调度
            # 默认节奏: [0.90, 0.82, 0.71, 0.56, 0.20, 0.0]
            # 特点: 首步 0.08 小步平滑缓冲保护潜空间，末步 0.20->0.0 完美复刻官方 8+1 经典收敛
            # ==================================================================
            custom_s2_env = os.environ.get("MINIMAX_H3_TWO_STAGE_CUSTOM_SIGMAS", "").strip()
            if custom_s2_env:
                try:
                    s2_sigmas_v = [float(x.strip()) for x in custom_s2_env.split(",") if x.strip()]
                    if s2_sigmas_v[-1] != 0.0:
                        s2_sigmas_v.append(0.0)
                except Exception:
                    s2_sigmas_v = [0.90, 0.82, 0.71, 0.56, 0.20, 0.0]
            else:
                s2_sigmas_v = [0.90, 0.82, 0.71, 0.56, 0.20, 0.0]

            stage2_steps = len(s2_sigmas_v) - 1

            # 注入二采初始高斯噪声 (与二采调度起点对齐)
            custom_sigma_env = os.environ.get("MINIMAX_H3_TWO_STAGE_NOISE_SIGMA", "").strip()
            sigma_refine_v = float(custom_sigma_env) if custom_sigma_env else s2_sigmas_v[0]
            s2_sigmas_v[0] = sigma_refine_v

            # 对应音频调度：按 Flow 变换时间映射对齐
            s2_sigmas_a = []
            for sv in s2_sigmas_v:
                if sv <= 0.0:
                    s2_sigmas_a.append(0.0)
                else:
                    t_val = sv / (v_shift - (v_shift - 1.0) * sv)
                    sa_val = float(torch.clamp(
                        torch.tensor((a_shift * t_val) / (1.0 + (a_shift - 1.0) * t_val)),
                        0.0,
                        1.0,
                    ).item())
                    s2_sigmas_a.append(sa_val)
            s2_sigmas_a[-1] = 0.0
            sigma_refine_a = s2_sigmas_a[0]

            # 跨卡同步生成高斯噪声
            noise_seed = (int(seed) + stage1_steps * 10007) & 0x7FFFFFFF
            gen_noise = torch.Generator(device="cpu").manual_seed(noise_seed)
            target_noise = torch.randn(
                new_video_target.shape,
                generator=gen_noise,
                dtype=torch.float32,
            ).to(device=device, dtype=new_video_target.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(target_noise, src=0)

            # 视频加噪: (1 - sigma) * x0 + sigma * noise
            new_video_target = (1.0 - sigma_refine_v) * new_video_target + sigma_refine_v * target_noise

            # 【首尾帧加噪掩码与真值锁死 (Keyframe Preserving Mask)】
            # 方案 A: 默认不进行单帧硬锁死，消除首帧与第二帧的阶梯跳动
            if lock_keyframes and raw_keyframe_rows is not None and keyframe_frame_indices:
                frame_rows = (new_h // 2) * (new_w // 2)
                for blk_idx, frame_idx in enumerate(keyframe_frame_indices):
                    src_slice = slice(blk_idx * frame_rows, (blk_idx + 1) * frame_rows)
                    clean_cond_frame = raw_keyframe_rows[src_slice].to(
                        device=device, dtype=new_video_target.dtype
                    )
                    if frame_idx == 0:
                        new_video_target[0:frame_rows] = clean_cond_frame
                        _log_master(
                            "    [*] [首尾帧保护] 视频第 0 帧已锁定为 100% 原始高清真值潜空间 (免受噪声污染)！\n",
                            flush=True,
                        )
                    elif frame_idx == -1 or (
                        keyframe_frame_count and frame_idx == keyframe_frame_count - 1
                    ):
                        dst_slice = slice((latent_t - 1) * frame_rows, latent_t * frame_rows)
                        new_video_target[dst_slice] = clean_cond_frame
                        _log_master(
                            "    [*] [首尾帧保护] 视频末帧已锁定为 100% 原始高清真值潜空间 (免受噪声污染)！\n",
                            flush=True,
                        )
            elif not lock_keyframes:
                _log_master(
                    "    [*] [方案A-时序平滑] 二采全视频统一加噪，不施加单帧人工硬掩码，杜绝首帧阶跃撕裂。\n",
                    flush=True,
                )

            # 音频同步加噪: (1 - sigma_a) * audio_x0 + sigma_a * audio_noise
            gen_audio_noise = torch.Generator(device="cpu").manual_seed((noise_seed + 101) & 0x7FFFFFFF)
            target_audio_noise = torch.randn(
                audio_rows[audio_target_slice].shape,
                generator=gen_audio_noise,
                dtype=torch.float32,
            ).to(device=device, dtype=audio_rows.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(target_audio_noise, src=0)
            audio_rows[audio_target_slice] = (
                (1.0 - sigma_refine_a) * audio_rows[audio_target_slice]
                + sigma_refine_a * target_audio_noise
            )

            # Dump 加噪后的 Xt
            if output_state is not None and os.environ.get(
                "MINIMAX_H3_DUMP_STAGE1_X0", ""
            ).strip() not in ("", "0", "false", "False"):
                output_state["stage2_noised_xt_latents"] = minimax_h3_unpatchify_video_tokens(
                    new_video_target.detach(),
                    latent_shape=[latent_t, new_h // 2, new_w // 2, 24],
                    patch_size=[1, 2, 2],
                ).contiguous()
                output_state["stage2_noised_xt_latent_hw"] = (int(new_h), int(new_w))

            _log_master(
                f"    [4/4] 注入二采初始噪声 (视频 sigma={sigma_refine_v:.4f}, 音频 sigma={sigma_refine_a:.4f}, "
                f"seed={noise_seed})\n"
                f"{'='*75}\n",
                flush=True,
            )

            # 条件网格对齐
            transition_cond = cond_anchor
            if stage2_layout_cond_rows is not None:
                transition_cond = stage2_layout_cond_rows.to(
                    device=device, dtype=torch.float32
                )
            elif stage2_keyframe_cond_rows is not None:
                src_h, src_w = stage2_keyframe_latent_hw
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
                    minimax_h3_resize_video_token_rows,
                )
                transition_cond = minimax_h3_resize_video_token_rows(
                    stage2_keyframe_cond_rows,
                    src_latent_h=src_h,
                    src_latent_w=src_w,
                    dst_latent_h=new_h,
                    dst_latent_w=new_w,
                ).to(device=device, dtype=torch.float32)

            new_video_rows = torch.zeros(
                int(positive.img_pos.shape[0]),
                new_video_target.shape[1],
                dtype=torch.float32,
                device=device,
            )
            if transition_cond is not None:
                new_video_rows.index_copy_(0, positive.cond_row_idx, transition_cond)
            new_video_rows[video_target_slice] = new_video_target
            video_rows = new_video_rows
            video_denoised_scratch = torch.empty_like(video_rows[video_target_slice])

            # ==================================================================
            # 【Stage 2 Schedule 应用】直接加载 X1 优化版非线性精修调度
            # ==================================================================
            sigmas_video = s2_sigmas_v
            sigmas_audio = s2_sigmas_a
            video_timesteps = [1.0 - s for s in sigmas_video[:-1]]
            audio_timesteps = [1.0 - s for s in sigmas_audio[:-1]]
            video_step_t = torch.tensor(video_timesteps, dtype=torch.float32, device=device)
            audio_step_t = torch.tensor(audio_timesteps, dtype=torch.float32, device=device)

            timestep_plan = positive.prepare_timestep_plan(
                video_timesteps=video_timesteps,
                audio_timesteps=audio_timesteps,
                imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
                audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
            )
            adaln_plan_slots = model.prepare_adaln_plans([entry[0] for entry in timestep_plan])

            video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
            audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
            video_sigma_ratios = video_sigmas[1:] / video_sigmas[:-1]
            audio_sigma_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
            video_sigma_t = 1.0 - video_step_t
            audio_sigma_t = 1.0 - audio_step_t
            video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
            audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios

            if sigma_refine_v <= 0.0:
                _log_master(
                    f"\n{'='*75}\n"
                    f">>> [TwoStage-FullX0] MINIMAX_H3_TWO_STAGE_NOISE_SIGMA=0: 完整去噪 + 3D 放大直接出片，跳过二采加噪！\n"
                    f"{'='*75}\n",
                    flush=True,
                )
            else:
                _log_master(
                    f">>> [TwoStage-FullX0] 启动 Stage 2/2: 二采 768P ({new_w*16}x{new_h*16}) 高清精修 (共 {stage2_steps} 步)\n"
                    f"    [+] Schedule: video_sigmas: {[round(x, 4) for x in s2_sigmas_v]}\n"
                    f"{'='*75}",
                    flush=True,
                )
                with tqdm(
                    total=stage2_steps,
                    desc=f"[Stage 2/2 二采 高清精修 ({new_h*16}P)]",
                    dynamic_ncols=True,
                    disable=not is_master,
                ) as pbar2:
                    for step in range(0, stage2_steps):
                        _execute_step(step)
                        # 【首尾帧每步真值锚定】方案 A 默认关闭，让 DiT 模型在时序注意力下自主平滑演进
                        if lock_keyframes and raw_keyframe_rows is not None and keyframe_frame_indices:
                            frame_rows = (new_h // 2) * (new_w // 2)
                            target_t = video_rows[video_target_slice]
                            for blk_idx, frame_idx in enumerate(keyframe_frame_indices):
                                src_slice = slice(blk_idx * frame_rows, (blk_idx + 1) * frame_rows)
                                clean_cond_frame = raw_keyframe_rows[src_slice].to(
                                    device=device, dtype=target_t.dtype
                                )
                                if frame_idx == 0:
                                    target_t[0:frame_rows] = clean_cond_frame
                                elif frame_idx == -1 or (
                                    keyframe_frame_count and frame_idx == keyframe_frame_count - 1
                                ):
                                    dst_slice = slice((latent_t - 1) * frame_rows, latent_t * frame_rows)
                                    target_t[dst_slice] = clean_cond_frame
                        if is_master:
                            pbar2.update(1)

    if output_state is not None:
        output_state["positive"] = positive
        if current_latent_dims is not None:
            output_state["latent_h"] = current_latent_dims[1]
            output_state["latent_w"] = current_latent_dims[2]

    return video_rows, audio_rows


__all__ = [
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MINIMAX_H3_VIDEO_ROW_WIDTH",
    "MiniMaxH3DenoiseBranch",
    "minimax_h3_denoise_loop",
]
