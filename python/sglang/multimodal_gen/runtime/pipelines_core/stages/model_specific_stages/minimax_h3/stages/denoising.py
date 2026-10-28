# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 denoise sink for packed-token DiT stepping, CFG-distilled
single-branch execution, and payload validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache, partial
from typing import Any

import torch

from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    disable_cache_on_transformer,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_CONFIG,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)

_REF2VA_VIDEO_CHAINS = {
    "video.reference_preserve",
    "video_audio.reference_preserve",
}


@lru_cache(maxsize=1)
def _denoise_total_steps_publisher():
    """Resolve the sparge_attn schedule-length publisher, or a no-op.

    Only sparge_attn's ``skip_last_steps`` consumes the schedule length.
    Resolving it lazily and degrading to ``nullcontext`` keeps every other
    backend on exactly the path it had before, and mirrors how the model
    publishes modality tags in ``minimax_h3.py``.
    """
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.sparge_attn import (
            sparge_denoise_total_steps,
        )

        return sparge_denoise_total_steps
    except Exception:  # pragma: no cover - defensive
        import contextlib

        return lambda _total: contextlib.nullcontext()


def _denoise_total_steps_ctx(total: int):
    return _denoise_total_steps_publisher()(total)


def minimax_h3_condition_noise_aug(sampling: Any) -> tuple[float, float]:
    """Resolve condition timesteps using the model defaults."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )

    imgvid_noise_aug = getattr(
        sampling,
        "imgvid_cond_noise_aug_for_inference",
        None,
    )
    if imgvid_noise_aug is None:
        # The model default uses imgvid noise aug 0.999. A request selecting
        # imgvid noise aug 1.0 still overrides this.
        imgvid_noise_aug = MINIMAX_H3_IMGVID_COND_TIMESTEP
    audio_noise_aug = getattr(
        sampling,
        "audio_cond_noise_aug_for_inference",
        None,
    )
    if audio_noise_aug is None:
        audio_noise_aug = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP
    return float(imgvid_noise_aug), float(audio_noise_aug)


def _validate_fl2va_keyframe_payload(plan: Any, keyframe: Any) -> None:
    """Reject stale/middle/reordered keyframe payloads at the DiT sink."""

    task = None if plan is None else str(plan.task)
    if task != "fl2va":
        if keyframe is not None:
            raise ValueError(
                "keyframe condition rows are only valid for plan.task='fl2va'"
            )
        return
    if not isinstance(keyframe, Mapping):
        raise ValueError("fl2va denoising requires encoded keyframe condition rows")

    semantic_indices = tuple(keyframe.get("semantic_frame_indices") or ())
    if semantic_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "fl2va denoising requires semantic_frame_indices in "
            f"{MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, "
            f"got {semantic_indices!r}"
        )
    frame_count = keyframe.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("fl2va keyframe payload requires an integer frame_count")
    if frame_count <= 1:
        raise ValueError("fl2va keyframe payload frame_count must be greater than one")
    pixel_indices = keyframe.get("pixel_frame_indices")
    expected_pixel_indices = [
        frame_count - 1 if index == -1 else index for index in semantic_indices
    ]
    if pixel_indices != expected_pixel_indices:
        raise ValueError(
            "fl2va denoising requires pixel_frame_indices resolved from the "
            "semantic anchors, "
            f"got {pixel_indices!r} for frame_count={frame_count}"
        )

    entries = keyframe.get("keyframes")
    if (
        not isinstance(entries, list)
        or len(entries) != len(semantic_indices)
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        raise ValueError(
            "fl2va denoising requires one encoded keyframe per semantic anchor"
        )
    if [entry.get("frame_index") for entry in entries] != list(semantic_indices):
        raise ValueError("fl2va encoded keyframes must remain in semantic anchor order")
    if [
        entry.get("resolved_frame_index") for entry in entries
    ] != expected_pixel_indices:
        raise ValueError(
            "fl2va encoded keyframes must carry matching resolved_frame_index values"
        )

    latent_h = int(keyframe.get("latent_h") or 0)
    latent_w = int(keyframe.get("latent_w") or 0)
    rows = keyframe.get("rows")
    expected_rows = len(semantic_indices) * (latent_h // 2) * (latent_w // 2)
    if (
        latent_h <= 0
        or latent_w <= 0
        or not isinstance(rows, torch.Tensor)
        or int(rows.shape[0]) != expected_rows
    ):
        actual_rows = None if not isinstance(rows, torch.Tensor) else int(rows.shape[0])
        raise ValueError(
            "fl2va encoded keyframe rows do not match target-canvas blocks: "
            f"expected={expected_rows}, actual={actual_rows}"
        )


def _imgvid_condition_shapes(
    *,
    ref2va_blocks: list[dict[str, int | str]] | None,
    keyframe: Any,
    is_ref2va: bool,
) -> list[tuple[int, int, int]]:
    """Return visual-condition ``(T,H,W)`` in packed anchor-row order."""

    if ref2va_blocks is not None:
        shapes = []
        for block in ref2va_blocks:
            kind = str(block["kind"])
            if kind == "image":
                shapes.append((1, int(block["latent_h"]), int(block["latent_w"])))
            elif kind in {"video", "video_audio"}:
                shapes.append(
                    (
                        int(block["latent_t"]),
                        int(block["latent_h"]),
                        int(block["latent_w"]),
                    )
                )
        return shapes

    if is_ref2va:
        # ref2va always carries a resolved plan, so ordered blocks are
        # supplied above; reaching here indicates an upstream bug.
        raise ValueError("ref2va visual-condition shapes require ordered blocks")

    if not isinstance(keyframe, Mapping):
        return []
    entries = keyframe.get("keyframes")
    if isinstance(entries, list) and entries:
        return [
            (1, int(entry["latent_h"]), int(entry["latent_w"])) for entry in entries
        ]

    latent_h = int(keyframe["latent_h"])
    latent_w = int(keyframe["latent_w"])
    frame_rows = (latent_h // 2) * (latent_w // 2)
    rows = keyframe["rows"]
    if frame_rows <= 0 or int(rows.shape[0]) % frame_rows:
        raise ValueError(
            "legacy keyframe rows cannot be split into visual-condition frames"
        )
    return [(1, latent_h, latent_w)] * (int(rows.shape[0]) // frame_rows)


def _ref2va_payload_entry(
    payload: Any,
    *,
    list_key: str,
    condition_index: int,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} is required for ref2va condition rows")
    entries = payload.get(list_key)
    if isinstance(entries, list):
        for entry in entries:
            if (
                isinstance(entry, Mapping)
                and entry.get("condition_index") is not None
                and int(entry["condition_index"]) == int(condition_index)
            ):
                return entry
        if len(entries) == 1 and isinstance(entries[0], Mapping):
            return entries[0]
        raise ValueError(
            f"{path}.{list_key} missing entry for condition_index={condition_index}"
        )
    if payload.get("condition_index") is None or int(payload["condition_index"]) == int(
        condition_index
    ):
        return payload
    raise ValueError(
        f"{path}.{list_key} missing entry for condition_index={condition_index}"
    )


def _cat_optional(rows: list[torch.Tensor]) -> torch.Tensor | None:
    if not rows:
        return None
    return rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)


def _even_half_latent_dim(dim: int, scale_factor: float) -> int:
    """Match draft-canvas rounding: /scale_factor, then even for 2x2 patchify."""

    return max(2, int(round(int(dim) / float(scale_factor))) // 2 * 2)


def _halve_ref2va_visual_conditions(
    blocks: list[dict[str, int | str]],
    visual_rows: torch.Tensor | None,
    *,
    scale_factor: float,
) -> tuple[list[dict[str, int | str]], torch.Tensor | None, list[str]]:
    """Spatially halve ref2va image/video tokens for two-stage draft denoise.

    Audio blocks are left unchanged. Each visual block keeps its own temporal
    length; only latent H/W follow the generation-canvas scale_factor.
    """

    if visual_rows is None:
        return [dict(block) for block in blocks], None, []

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_resize_video_token_rows,
    )

    cursor = 0
    new_blocks: list[dict[str, int | str]] = []
    new_parts: list[torch.Tensor] = []
    notes: list[str] = []
    n_visual = int(visual_rows.shape[0])
    for block in blocks:
        new_block = dict(block)
        kind = str(block["kind"])
        if kind == "image":
            src_h = int(block["latent_h"])
            src_w = int(block["latent_w"])
            n_rows = (src_h // 2) * (src_w // 2)
            chunk = visual_rows[cursor : cursor + n_rows]
            cursor += n_rows
            dst_h = _even_half_latent_dim(src_h, scale_factor)
            dst_w = _even_half_latent_dim(src_w, scale_factor)
            if (dst_h, dst_w) != (src_h, src_w):
                chunk = minimax_h3_resize_video_token_rows(
                    chunk,
                    src_latent_h=src_h,
                    src_latent_w=src_w,
                    dst_latent_h=dst_h,
                    dst_latent_w=dst_w,
                )
            new_block["latent_h"] = dst_h
            new_block["latent_w"] = dst_w
            new_parts.append(chunk)
            notes.append(
                f"image {src_h}x{src_w} -> {dst_h}x{dst_w} "
                f"({n_rows} -> {int(chunk.shape[0])} rows)"
            )
        elif kind in {"video", "video_audio"}:
            src_h = int(block["latent_h"])
            src_w = int(block["latent_w"])
            latent_t = int(block["latent_t"])
            n_rows = latent_t * (src_h // 2) * (src_w // 2)
            chunk = visual_rows[cursor : cursor + n_rows]
            cursor += n_rows
            dst_h = _even_half_latent_dim(src_h, scale_factor)
            dst_w = _even_half_latent_dim(src_w, scale_factor)
            if (dst_h, dst_w) != (src_h, src_w):
                chunk = minimax_h3_resize_video_token_rows(
                    chunk,
                    src_latent_h=src_h,
                    src_latent_w=src_w,
                    dst_latent_h=dst_h,
                    dst_latent_w=dst_w,
                )
            new_block["latent_h"] = dst_h
            new_block["latent_w"] = dst_w
            new_parts.append(chunk)
            notes.append(
                f"{kind} T={latent_t} {src_h}x{src_w} -> {dst_h}x{dst_w} "
                f"({n_rows} -> {int(chunk.shape[0])} rows)"
            )
        new_blocks.append(new_block)
    if cursor != n_visual:
        raise ValueError(
            "ref2va visual rows did not match block layout while halving for "
            f"draft denoise: consumed={cursor} actual={n_visual}"
        )
    halved = _cat_optional(new_parts)
    return new_blocks, halved, notes


def _ref2va_ordered_blocks_and_rows(
    *,
    plan: Any,
    ref_image: Any,
    ref_audio: Any,
    ref_video: Any,
) -> tuple[list[dict[str, int | str]], torch.Tensor | None, torch.Tensor | None]:
    blocks: list[dict[str, int | str]] = []
    visual_rows: list[torch.Tensor] = []
    audio_rows: list[torch.Tensor] = []
    for material in plan.materials:
        chain = str(material.material_chain)
        condition_index = int(material.condition_index)
        if chain == "image.reference_preserve":
            entry = _ref2va_payload_entry(
                ref_image,
                list_key="images",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_image_rows",
            )
            blocks.append(
                {
                    "kind": "image",
                    "latent_h": int(entry["latent_h"]),
                    "latent_w": int(entry["latent_w"]),
                }
            )
            visual_rows.append(entry["rows"])
        elif chain == "audio":
            entry = _ref2va_payload_entry(
                ref_audio,
                list_key="audios",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_audio_rows",
            )
            ref_audio_t = int(entry["ref_audio_t"])
            blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
            if ref_audio_t > 0:
                audio_rows.append(entry["rows"])
        elif chain in _REF2VA_VIDEO_CHAINS:
            video_entry = _ref2va_payload_entry(
                ref_video,
                list_key="videos",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_video_rows",
            )
            audio_entry = _ref2va_payload_entry(
                ref_audio,
                list_key="audios",
                condition_index=condition_index,
                path="batch.extra.minimax_h3_reference_audio_rows",
            )
            ref_audio_t = int(audio_entry["ref_audio_t"])
            blocks.append(
                {
                    "kind": (
                        "video_audio"
                        if chain == "video_audio.reference_preserve"
                        else "video"
                    ),
                    "ref_audio_t": ref_audio_t,
                    "latent_t": int(video_entry["latent_t"]),
                    "latent_h": int(video_entry["latent_h"]),
                    "latent_w": int(video_entry["latent_w"]),
                }
            )
            visual_rows.append(video_entry["rows"])
            if ref_audio_t > 0:
                audio_rows.append(audio_entry["rows"])
        else:
            raise ValueError(f"unsupported ref2va material chain {chain!r}")
    return blocks, _cat_optional(visual_rows), _cat_optional(audio_rows)


def _resolve_denoise_model(
    transformer: Any,
    device: torch.device,
    *,
    placement_managed: bool = False,
) -> Any:
    """Resolve the DiT module and place it for denoise.

    ComponentManager and FSDP own their device placement; move only unmanaged
    plain modules.
    """
    model = getattr(transformer, "model", transformer)
    if placement_managed or is_fsdp_managed_module(model):
        if model.training:
            model.eval()
        return model
    return model.to(device).eval()


_REFINED_PROMPT_EMBEDS_KEY = "_minimax_h3_refined_prompt_embeds"


def _precompute_refined_prompt_embeds(
    model: Any,
    positive: Any,
    *,
    device: torch.device,
    shared_conditioning: dict[str, Any] | None = None,
) -> bool:
    """Move request-static text refinement out of the denoise hot loop."""
    refine = getattr(model, "refine_prompt_embeds", None)
    if not callable(refine):
        return False

    static_kwargs = positive.static_kwargs
    prompt_embeds = static_kwargs["prompt_embeds"]
    refined = (
        shared_conditioning.get(_REFINED_PROMPT_EMBEDS_KEY)
        if shared_conditioning is not None
        else None
    )
    if refined is None:
        refiner_params = static_kwargs["refiner_packed_seq_params"]
        if isinstance(refiner_params, dict):
            refiner_cu = refiner_params["cu_seqlens_q"]
        else:
            refiner_cu = refiner_params.cu_seqlens_q
        with torch.inference_mode():
            refined = refine(
                prompt_embeds,
                refiner_cu,
                device=device,
            )
    if not torch.is_tensor(refined):
        raise TypeError("MiniMax H3 refine_prompt_embeds must return a torch.Tensor")
    if int(refined.shape[0]) != int(prompt_embeds.shape[0]):
        raise ValueError(
            "MiniMax H3 refined prompt row count changed: "
            f"{int(prompt_embeds.shape[0])} -> {int(refined.shape[0])}"
        )
    if shared_conditioning is not None:
        shared_conditioning.setdefault(_REFINED_PROMPT_EMBEDS_KEY, refined)
    static_kwargs["prompt_embeds"] = refined
    static_kwargs["refined_prompt_embeds_length"] = int(refined.shape[0])
    return True


def _precompute_rope_cache(
    model: Any,
    positive: Any,
    *,
    device: torch.device,
) -> bool:
    """Move request-static RoPE construction out of the denoise hot loop."""
    build = getattr(model, "build_rope_cache", None)
    if not callable(build):
        return False
    static_kwargs = positive.static_kwargs
    with torch.inference_mode():
        static_kwargs["rope_cache"] = build(
            static_kwargs["img_position_ids"],
            device=device,
        )
    return True


class MiniMaxH3DenoisingStage(DenoisingStage):
    def __init__(self, transformer, pipeline=None) -> None:
        super().__init__(
            transformer=transformer,
            scheduler=None,
            pipeline=pipeline,
        )
        self._minimax_h3_quality = "lossless"
        self._minimax_h3_cache_mode: str | None = None

    def _owns_compile_warmup_lifecycle(self) -> bool:
        return True

    def _cache_dit_requested(self) -> bool:
        return (
            getattr(self, "_minimax_h3_quality", "lossless") == "high"
            or super()._cache_dit_requested()
        )

    def _maybe_enable_cache_dit(
        self, num_inference_steps: int | tuple[int, int], batch: Req
    ) -> None:
        quality = getattr(batch.sampling_params, "quality", "lossless")
        explicit_fields = getattr(batch.sampling_params, "_explicit_fields", ())
        generic_requested = (
            super()._cache_dit_requested() and "quality" not in explicit_fields
        )
        desired_mode = (
            "high" if quality == "high" else ("generic" if generic_requested else None)
        )
        current_mode = getattr(self, "_minimax_h3_cache_mode", None)
        self._minimax_h3_quality = quality

        if self.server_args.enable_breakable_cuda_graph:
            if desired_mode is not None:
                super()._maybe_enable_cache_dit(num_inference_steps, batch)
            return

        # H3 is monolithic-only, and the scheduler executes one worker batch at
        # a time. Combined with `quality` in the dynamic-batch signature, this
        # makes the process-wide hook transition safe at this batch boundary.
        if self._cache_dit_enabled and current_mode != desired_mode:
            # Unmount first: the blocks must stay preserved for as long as
            # Cache-DiT still holds references to their inputs. Settle the state
            # fields before restoring the in-place path, so a failure there
            # costs throughput rather than leaving the stage inconsistent.
            self.transformer = disable_cache_on_transformer(self.transformer)
            self._cache_dit_enabled = False
            self._cached_num_steps = None
            self._minimax_h3_cache_mode = None
            self._set_cache_dit_input_preservation(False)

        if desired_mode is None:
            return

        # Arm before delegating whenever this H3 stage requests caching,
        # without predicting whether the parent will accept the mount.
        # cache_dit.enable_cache swaps `blocks` for a single CachedBlocks
        # wrapper, so the real blocks are only reachable beforehand -- and a
        # wrong prediction would hand Cache-DiT unpreserved blocks, whose
        # residuals read as zero, which is silent. Arming is one boolean per
        # block and nothing runs before the parent decides, so guessing
        # conservatively costs approximately nothing.
        was_enabled = self._cache_dit_enabled
        if not was_enabled:
            self._set_cache_dit_input_preservation(True)
        try:
            super()._maybe_enable_cache_dit(num_inference_steps, batch)
        except Exception:
            if not was_enabled:
                self._disarm_after_failed_mount()
            raise

        if self._cache_dit_enabled:
            self._minimax_h3_cache_mode = desired_mode
        elif not was_enabled:
            # The parent declined to mount, for example because breakable
            # CUDA graphs are enabled or this is an ordinary warmup. Nothing
            # holds the block inputs, so go back to the in-place path.
            self._set_cache_dit_input_preservation(False)

    def _disarm_after_failed_mount(self) -> None:
        """Restore the in-place path only once Cache-DiT is confirmed gone.

        cache_dit.enable_cache swaps the block list before the rest of the mount
        runs, so a later failure can leave caching attached. Disarming then
        would hand Cache-DiT unpreserved blocks and silently reproduce the
        zero-residual bug, so if the unmount does not succeed we stay armed and
        pay throughput instead.
        """
        try:
            self.transformer = disable_cache_on_transformer(self.transformer)
        except Exception:
            logger.warning(
                "Could not unmount Cache-DiT after a failed mount; leaving "
                "MiniMax-H3 input preservation on",
                exc_info=True,
            )
            return
        # The parent may have flipped these before failing; leaving them set
        # would send the next request down the refresh path with nothing
        # mounted.
        self._cache_dit_enabled = False
        self._cached_num_steps = None
        self._minimax_h3_cache_mode = None
        self._set_cache_dit_input_preservation(False)

    def _set_cache_dit_input_preservation(self, enabled: bool) -> None:
        """Flip input preservation on the H3 model, failing closed.

        Skipping this silently would put us back where this stage started: cache
        mounted, residuals reading as zero, no hits and nothing logged. If the
        model cannot be reached, that is a wiring bug and should surface here.
        """
        model = self.transformer
        for _ in range(4):  # unwrap compile/parallel wrappers, if any
            if hasattr(model, "set_cache_dit_input_preservation"):
                break
            inner = getattr(model, "_orig_mod", None) or getattr(model, "module", None)
            if inner is None:
                break
            model = inner
        setter = getattr(model, "set_cache_dit_input_preservation", None)
        if not callable(setter):
            raise TypeError(
                "MiniMax-H3 Cache-DiT requires set_cache_dit_input_preservation() "
                f"on the transformer, but {type(self.transformer).__name__} does "
                "not expose it"
            )
        setter(enabled)

    def _cache_dit_scm_masks(
        self, primary_num_steps: int, secondary_num_steps: int | None = None
    ) -> tuple[str, str, list[int] | None, list[int] | None]:
        if getattr(self, "_minimax_h3_quality", "lossless") == "high":
            return "none", "dynamic", None, None
        return super()._cache_dit_scm_masks(primary_num_steps, secondary_num_steps)

    def _build_cache_dit_config(
        self,
        num_inference_steps: int,
        steps_computation_mask: list[int] | None,
        scm_policy: str,
        *,
        secondary: bool = False,
    ) -> CacheDitConfig:
        if secondary or getattr(self, "_minimax_h3_quality", "lossless") != "high":
            return super()._build_cache_dit_config(
                num_inference_steps,
                steps_computation_mask,
                scm_policy,
                secondary=secondary,
            )
        warmup, threshold, max_cached = MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_CONFIG
        return CacheDitConfig(
            enabled=True,
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            max_warmup_steps=warmup,
            residual_diff_threshold=threshold,
            max_continuous_cached_steps=max_cached,
            enable_taylorseer=False,
            taylorseer_order=1,
            num_inference_steps=num_inference_steps,
            steps_computation_mask=steps_computation_mask,
            steps_computation_policy=scm_policy,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with self._offload_for_torch_compile_warmup(batch):
            return self._forward_native(batch, server_args)

    def _forward_native(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
            MINIMAX_H3_SIGMAS_EXTRA_KEY,
            MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
        )

        if MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY in batch.extra:
            for required in (
                MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
                MINIMAX_H3_SIGMAS_EXTRA_KEY,
            ):
                if required not in batch.extra:
                    raise ValueError(
                        f"direct full-loop denoise requires batch.extra[{required!r}]"
                    )
            self._run_full_loop(batch, server_args)
            return batch
        if (
            batch.latents is not None
            or batch.audio_latents is not None
            or batch.timestep is not None
            or batch.timesteps is not None
        ):
            raise NotImplementedError(
                "MiniMaxH3DenoisingStage requires the canonical direct pipeline "
                "to populate text embeddings, denoise state, and sigma schedules."
            )
        return batch

    def _run_full_loop(self, batch: Req, server_args: ServerArgs) -> None:
        """Assemble the cfg-distilled positive input and run the full loop.

        The heavy lifting is decomposed into per-phase helpers:
        context/payload resolution, condition-row assembly, packed-layout
        construction, condition noise augmentation, initial-row expansion,
        the denoise loop itself, and output publication.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
            MiniMaxH3DenoiseBranch,
            minimax_h3_denoise_loop,
        )

        ctx = _resolve_full_loop_context(batch)

        if not torch.cuda.is_available():
            raise RuntimeError("MiniMax H3 full-loop denoise requires CUDA")
        device = torch.device("cuda")
        sigmas_video = [float(v) for v in ctx.sigmas["video"]]
        self._maybe_enable_cache_dit_and_torch_compile(
            len(sigmas_video) - 1,
            batch,
        )

        _assemble_condition_rows(ctx)

        emb = ctx.embeddings["positive"]
        sampling = batch.sampling_params
        imgvid_noise_aug, audio_noise_aug = minimax_h3_condition_noise_aug(sampling)
        _apply_condition_noise_aug(
            ctx,
            sampling=sampling,
            imgvid_noise_aug=imgvid_noise_aug,
            audio_noise_aug=audio_noise_aug,
        )

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.latent_upscaler_3d import (
            TwoStageDenoiseManager,
            is_two_stage_enabled,
        )

        two_stage_active = is_two_stage_enabled(batch=batch, server_args=server_args)
        split_step = 4
        scale_factor = 2.0
        upscaler_path = None
        if two_stage_active:
            sp = getattr(batch, "sampling_params", None)
            if sp is not None:
                split_step = int(getattr(sp, "two_stage_split_step", 4) or 4)
                scale_factor = float(getattr(sp, "two_stage_scale_factor", 2.0) or 2.0)
                upscaler_path = getattr(sp, "two_stage_upscaler_path", None)
            elif isinstance(batch.extra, dict):
                split_step = int(batch.extra.get("two_stage_split_step", 4) or 4)
                scale_factor = float(batch.extra.get("two_stage_scale_factor", 2.0) or 2.0)
                upscaler_path = batch.extra.get("two_stage_upscaler_path")

        stage2_ref_blocks = None
        stage2_ref_cond = None
        if (
            two_stage_active
            and ctx.is_ref2va
            and ctx.cond_rows is not None
            and ctx.ref2va_positive_blocks is not None
        ):
            stage2_ref_blocks = [dict(block) for block in ctx.ref2va_positive_blocks]
            stage2_ref_cond = ctx.cond_rows
            ctx.ref2va_positive_blocks, ctx.cond_rows, ref_notes = (
                _halve_ref2va_visual_conditions(
                    ctx.ref2va_positive_blocks,
                    ctx.cond_rows,
                    scale_factor=scale_factor,
                )
            )
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_world_rank,
                world_group_is_initialized,
            )

            if not world_group_is_initialized() or get_world_rank() == 0:
                detail = "; ".join(ref_notes) if ref_notes else "unchanged"
                print(
                    ">>> [TwoStage] 参考条件对齐到一采 Draft 潜空间 "
                    f"(scale={scale_factor:g}x): {detail}",
                    flush=True,
                )

        packed = _build_packed_layout(ctx, emb)
        tags = packed["token_tags"]
        tags[packed["text_pos"].view(-1)] = (
            emb["text_token_tags"].view(-1).to(device=tags.device, dtype=torch.long)
        )

        placement_managed = self._component_residency_manager is not None
        if placement_managed:
            self._manage_dit_use_site(self.transformer, "transformer", batch)
        try:
            model = _resolve_denoise_model(
                self.transformer,
                device,
                placement_managed=placement_managed,
            )
            positive = MiniMaxH3DenoiseBranch(
                packed=packed,
                text_embeddings=emb["hidden_states"],
                token_tags=tags,
                device=device,
            )
            _precompute_refined_prompt_embeds(
                model,
                positive,
                device=device,
                shared_conditioning=emb,
            )
            _precompute_rope_cache(
                model,
                positive,
                device=device,
            )
            initial_video, initial_audio = _expand_initial_rows(ctx, positive)

            two_stage_mgr = None
            if two_stage_active:
                import os
                if not upscaler_path:
                    upscaler_path = os.environ.get("MINIMAX_H3_UPSCALER_PATH")

                model_dtype = torch.float32 if (upscaler_path and "fp32" in upscaler_path.lower()) else torch.float16
                mgr_kwargs = {
                    "enabled": True,
                    "split_step": split_step,
                    "scale_factor": scale_factor,
                    "device": device,
                    "dtype": model_dtype,
                }
                if upscaler_path:
                    mgr_kwargs["model_path"] = upscaler_path
                two_stage_mgr = TwoStageDenoiseManager(**mgr_kwargs)
                from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                    get_world_rank,
                    world_group_is_initialized,
                )
                is_main_rank = not world_group_is_initialized() or get_world_rank() == 0
                if is_main_rank:
                    print(
                        f"\n>>> [TwoStage] TwoStageDenoiseManager 已成功挂载 (转折步数={split_step}, 权重={upscaler_path})\n",
                        flush=True,
                    )

            def rebuild_layout_fn(new_h: int, new_w: int) -> tuple[Any, Any]:
                """【转折点核心回调】当视频潜空间从 Draft 放大至高清后，动态重构 Packed 序列与注意力元数据。"""
                if stage2_ref_blocks is not None:
                    ctx.ref2va_positive_blocks = stage2_ref_blocks
                    ctx.cond_rows = stage2_ref_cond
                ctx.latent_h = new_h
                ctx.latent_w = new_w
                new_packed = _build_packed_layout(ctx, emb)
                new_tags = new_packed["token_tags"]
                new_tags[new_packed["text_pos"].view(-1)] = (
                    emb["text_token_tags"]
                    .view(-1)
                    .to(device=new_tags.device, dtype=torch.long)
                )
                new_positive = MiniMaxH3DenoiseBranch(
                    packed=new_packed,
                    text_embeddings=emb["hidden_states"],
                    token_tags=new_tags,
                    device=device,
                )
                _precompute_refined_prompt_embeds(
                    model,
                    new_positive,
                    device=device,
                    shared_conditioning=emb,
                )
                _precompute_rope_cache(
                    model,
                    new_positive,
                    device=device,
                )
                return new_positive, None

            loop_output_state: dict[str, Any] = {}
            total_loop_steps = len(sigmas_video) - 1

            req_seed = 42
            if hasattr(batch, "sampling_params") and batch.sampling_params is not None:
                req_seed = getattr(batch.sampling_params, "seed", 42) or 42
            elif hasattr(ctx, "plan") and ctx.plan is not None:
                req_seed = getattr(ctx.plan, "seed", 42) or 42

            stage1_cond_rows = ctx.cond_rows
            stage2_cond_rows = None
            stage2_cond_hw = None
            if two_stage_mgr is not None and getattr(two_stage_mgr, "enabled", False):
                # 首帧按请求画布编码，一采布局却是半分辨率。优先使用像素级 Lanczos 编码的真实潜空间！
                if ctx.cond_rows is not None and ctx.keyframe is not None:
                    src_h = int(ctx.keyframe["latent_h"])
                    src_w = int(ctx.keyframe["latent_w"])
                    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                        get_world_rank,
                        world_group_is_initialized,
                    )
                    is_master_rank = not world_group_is_initialized() or get_world_rank() == 0

                    if "draft_rows" in ctx.keyframe and ctx.keyframe["draft_rows"] is not None:
                        stage1_cond_rows = ctx.keyframe["draft_rows"].to(
                            device=ctx.cond_rows.device, dtype=ctx.cond_rows.dtype
                        )
                        stage2_cond_rows = ctx.cond_rows
                        stage2_cond_hw = (src_h, src_w)
                        if is_master_rank:
                            print(
                                ">>> [TwoStage-PixelLanczos] 启用像素空间 Lanczos 原生编码的 Draft 首尾帧条件 "
                                f"({int(stage1_cond_rows.shape[0])} rows)，彻底杜绝 Latent 插值模糊！",
                                flush=True,
                            )
                    elif src_h != int(ctx.latent_h) or src_w != int(ctx.latent_w):
                        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
                            minimax_h3_resize_video_token_rows,
                        )

                        stage1_cond_rows = minimax_h3_resize_video_token_rows(
                            ctx.cond_rows,
                            src_latent_h=src_h,
                            src_latent_w=src_w,
                            dst_latent_h=int(ctx.latent_h),
                            dst_latent_w=int(ctx.latent_w),
                        )
                        stage2_cond_rows = ctx.cond_rows
                        stage2_cond_hw = (src_h, src_w)
                        if is_master_rank:
                            print(
                                ">>> [TwoStage] 首帧条件对齐到一采 Draft 潜空间 "
                                f"{src_h}x{src_w} -> {int(ctx.latent_h)}x{int(ctx.latent_w)} "
                                f"({int(ctx.cond_rows.shape[0])} -> {int(stage1_cond_rows.shape[0])} rows)",
                                flush=True,
                            )

            with (
                maybe_nvtx_range("denoising_loop", self.current_use_nvtx),
                _denoise_total_steps_ctx(total_loop_steps),
                self.progress_bar(
                    total=total_loop_steps,
                    batch=batch,
                    desc="minimax_h3 denoise",
                ) as progress_bar,
            ):

                def on_step(_step, _video_rows, _audio_rows):
                    progress_bar.update()
                    if not batch.is_warmup:
                        self.step_profile()

                video_rows, audio_rows = minimax_h3_denoise_loop(
                    model=model,
                    model_forward=partial(self._forward_dit, batch=batch),
                    positive=positive,
                    initial_video_rows=initial_video,
                    initial_audio_rows=initial_audio,
                    keyframe_cond_rows=stage1_cond_rows,
                    stage2_keyframe_cond_rows=stage2_cond_rows,
                    stage2_keyframe_latent_hw=stage2_cond_hw,
                    raw_keyframe_rows=ctx.keyframe["rows"] if ctx.keyframe is not None else None,
                    keyframe_frame_indices=ctx.keyframe_frame_indices if ctx.keyframe is not None else None,
                    keyframe_frame_count=ctx.keyframe_frame_count if ctx.keyframe is not None else None,
                    stage2_layout_cond_rows=stage2_ref_cond,
                    audio_ref_rows=ctx.audio_ref_rows,
                    sigmas_video=sigmas_video,
                    sigmas_audio=[float(v) for v in ctx.sigmas["audio"]],
                    device=device,
                    imgvid_cond_noise_aug_for_inference=float(imgvid_noise_aug),
                    audio_cond_noise_aug_for_inference=float(audio_noise_aug),
                    on_step=on_step,
                    step_profiler=partial(
                        self._profile_denoising_step,
                        batch=batch,
                    ),
                    two_stage_manager=two_stage_mgr,
                    rebuild_layout_fn=rebuild_layout_fn if two_stage_mgr else None,
                    current_latent_dims=(ctx.latent_t, ctx.latent_h, ctx.latent_w) if two_stage_mgr else None,
                    target_latent_hw=_two_stage_target_latent_hw(ctx) if two_stage_mgr else None,
                    output_state=loop_output_state,
                    seed=int(req_seed),
                )

            if "positive" in loop_output_state:
                positive = loop_output_state["positive"]
            if "latent_h" in loop_output_state:
                ctx.latent_h = loop_output_state["latent_h"]
            if "latent_w" in loop_output_state:
                ctx.latent_w = loop_output_state["latent_w"]
            if "stage1_x0_latents" in loop_output_state:
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
                    MINIMAX_H3_STAGE1_NOISY_XT_LATENT_HW_EXTRA_KEY,
                    MINIMAX_H3_STAGE1_NOISY_XT_LATENTS_EXTRA_KEY,
                    MINIMAX_H3_STAGE1_X0_LATENT_HW_EXTRA_KEY,
                    MINIMAX_H3_STAGE1_X0_LATENTS_EXTRA_KEY,
                    MINIMAX_H3_STAGE1_UPSCALED_X0_LATENT_HW_EXTRA_KEY,
                    MINIMAX_H3_STAGE1_UPSCALED_X0_LATENTS_EXTRA_KEY,
                    MINIMAX_H3_STAGE2_NOISED_XT_LATENT_HW_EXTRA_KEY,
                    MINIMAX_H3_STAGE2_NOISED_XT_LATENTS_EXTRA_KEY,
                )

                if "stage1_noisy_xt_latents" in loop_output_state:
                    batch.extra[MINIMAX_H3_STAGE1_NOISY_XT_LATENTS_EXTRA_KEY] = (
                        loop_output_state["stage1_noisy_xt_latents"]
                    )
                    batch.extra[MINIMAX_H3_STAGE1_NOISY_XT_LATENT_HW_EXTRA_KEY] = (
                        loop_output_state.get("stage1_noisy_xt_latent_hw")
                    )
                batch.extra[MINIMAX_H3_STAGE1_X0_LATENTS_EXTRA_KEY] = loop_output_state[
                    "stage1_x0_latents"
                ]
                batch.extra[MINIMAX_H3_STAGE1_X0_LATENT_HW_EXTRA_KEY] = (
                    loop_output_state.get("stage1_x0_latent_hw")
                )
                if "stage1_upscaled_x0_latents" in loop_output_state:
                    batch.extra[MINIMAX_H3_STAGE1_UPSCALED_X0_LATENTS_EXTRA_KEY] = (
                        loop_output_state["stage1_upscaled_x0_latents"]
                    )
                    batch.extra[MINIMAX_H3_STAGE1_UPSCALED_X0_LATENT_HW_EXTRA_KEY] = (
                        loop_output_state.get("stage1_upscaled_x0_latent_hw")
                    )
                if "stage2_noised_xt_latents" in loop_output_state:
                    batch.extra[MINIMAX_H3_STAGE2_NOISED_XT_LATENTS_EXTRA_KEY] = (
                        loop_output_state["stage2_noised_xt_latents"]
                    )
                    batch.extra[MINIMAX_H3_STAGE2_NOISED_XT_LATENT_HW_EXTRA_KEY] = (
                        loop_output_state.get("stage2_noised_xt_latent_hw")
                    )
        finally:
            self._finish_active_component_use()
        _publish_full_loop_outputs(
            ctx,
            batch=batch,
            positive=positive,
            video_rows=video_rows,
            audio_rows=audio_rows,
        )

    @contextmanager
    def _profile_denoising_step(self, step_index: int, *, batch: Req):
        with (
            maybe_nvtx_range(
                f"denoising_step_{step_index}",
                self.current_use_nvtx,
            ),
            StageProfiler(
                f"denoising_step_{step_index}",
                logger=logger,
                metrics=batch.metrics,
                perf_dump_path_provided=batch.perf_dump_path is not None,
                record_as_step=True,
            ),
        ):
            yield

    def _forward_dit(
        self,
        model: Any,
        call_kwargs: dict[str, Any],
        step_index: int,
        *,
        batch: Req,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route the custom full loop through the native denoising runner."""

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        with set_forward_context(
            current_timestep=step_index,
            attn_metadata=None,
            forward_batch=batch,
        ):
            runner = self._maybe_get_bcg_runner(model)
            if runner is None:
                return model(**call_kwargs)
            return self._bcg_run(runner, call_kwargs, model)

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "audio_latents",
            batch.audio_latents,
            [V.is_tensor, V.with_dims(3)],
        )
        return result


def _two_stage_target_latent_hw(ctx: "_FullLoopContext") -> tuple[int, int] | None:
    """二采超分必须落回的请求潜空间。没记下目标尺寸时保持超分器自己的对齐结果。"""

    state = ctx.state if isinstance(ctx.state, dict) else None
    if state is None:
        return None
    raw_h = state.get("target_latent_h")
    raw_w = state.get("target_latent_w")
    if raw_h is None or raw_w is None:
        return None
    target_h, target_w = int(raw_h), int(raw_w)
    if target_h <= 0 or target_w <= 0:
        return None
    if target_h % 2 or target_w % 2:
        target_h -= target_h % 2
        target_w -= target_w % 2
    if target_h <= 0 or target_w <= 0:
        return None
    return target_h, target_w


class _FullLoopContext:
    """Mutable per-request state threaded through the full-loop phases."""

    __slots__ = (
        "plan",
        "keyframe",
        "ref_image",
        "ref_audio",
        "ref_video",
        "is_ref2va",
        "embeddings",
        "state",
        "sigmas",
        "latent_t",
        "latent_h",
        "latent_w",
        "audio_t",
        "ref2va_positive_blocks",
        "cond_rows",
        "audio_ref_rows",
        "include_cond",
        "keyframe_frame_indices",
        "keyframe_frame_count",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, None)
        self.is_ref2va = False
        self.include_cond = False


def _resolve_full_loop_context(batch: Req) -> _FullLoopContext:
    """Read/validate extras and the denoise state into a loop context.

    Enforces the task-payload exclusivity rules (keyframe vs reference
    exclusivity) and cross-checks the resolved latent dims.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
        MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
        MINIMAX_H3_SIGMAS_EXTRA_KEY,
        MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        minimax_h3_plan_from_batch,
    )

    ctx = _FullLoopContext()
    ctx.embeddings = batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]
    ctx.state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
    ctx.sigmas = batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
    ctx.plan = minimax_h3_plan_from_batch(batch)
    ctx.keyframe = batch.extra.get(MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY)
    ctx.ref_image = batch.extra.get(MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY)
    ctx.ref_audio = batch.extra.get(MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY)
    ctx.ref_video = batch.extra.get(MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY)
    ctx.is_ref2va = (
        ctx.ref_image is not None
        or ctx.ref_audio is not None
        or ctx.ref_video is not None
    )
    if ctx.is_ref2va and ctx.keyframe is not None:
        raise ValueError("keyframe and reference extras are mutually exclusive")
    _validate_fl2va_keyframe_payload(ctx.plan, ctx.keyframe)

    ctx.latent_t = int(ctx.state["latent_t"])
    ctx.latent_h = int(ctx.state["latent_h"])
    ctx.latent_w = int(ctx.state["latent_w"])
    ctx.audio_t = int(ctx.state["audio_t"])
    return ctx


def _assemble_condition_rows(ctx: _FullLoopContext) -> None:
    """Populate cond/audio reference rows and keyframe metadata per task."""

    ctx.include_cond = (
        ctx.keyframe is not None
        or ctx.ref_image is not None
        or ctx.ref_video is not None
    )
    if ctx.is_ref2va:
        if ctx.plan is None:
            raise ValueError(
                "ref2va reference extras require a resolved plan; "
                "plan-less ref2va requests are unsupported"
            )
        ctx.ref2va_positive_blocks, ctx.cond_rows, ctx.audio_ref_rows = (
            _ref2va_ordered_blocks_and_rows(
                plan=ctx.plan,
                ref_image=ctx.ref_image,
                ref_audio=ctx.ref_audio,
                ref_video=ctx.ref_video,
            )
        )
        ctx.include_cond = ctx.cond_rows is not None
    else:
        ctx.cond_rows = ctx.keyframe["rows"] if ctx.include_cond else None
        ctx.audio_ref_rows = (
            ctx.ref_audio["rows"] if ctx.ref_audio is not None else None
        )
    if ctx.keyframe is not None:
        raw_indices = ctx.keyframe.get("semantic_frame_indices")
        ctx.keyframe_frame_indices = [int(v) for v in raw_indices]
        ctx.keyframe_frame_count = int(ctx.keyframe["frame_count"])


def _build_packed_layout(
    ctx: _FullLoopContext,
    emb: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build the per-task packed layout for the positive branch."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    if ctx.is_ref2va:
        if ctx.ref2va_positive_blocks is None:
            raise ValueError("ref2va ordered reference blocks missing")
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=int(emb["text_len"]),
            latent_t=ctx.latent_t,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
            audio_t=ctx.audio_t,
            ref_blocks=ctx.ref2va_positive_blocks,
        )
    else:
        packed = minimax_h3_packed_sequence(
            text_len=int(emb["text_len"]),
            latent_t=ctx.latent_t,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
            audio_t=ctx.audio_t,
            include_keyframe_cond=ctx.include_cond,
            keyframe_frame_indices=(
                ctx.keyframe_frame_indices if ctx.include_cond else None
            ),
            frame_count=ctx.keyframe_frame_count,
        )
    return packed


def _condition_audio_lengths(ctx: _FullLoopContext) -> list[int]:
    """Per-task condition audio T list for the noise-aug recipe."""

    condition_audio_t: list[int] = []
    if ctx.ref2va_positive_blocks is not None:
        for block in ctx.ref2va_positive_blocks:
            if str(block["kind"]) in {"audio", "video", "video_audio"}:
                ref_audio_t = int(block["ref_audio_t"])
                if ref_audio_t > 0:
                    condition_audio_t.append(ref_audio_t)
    elif isinstance(ctx.ref_audio, Mapping):
        entries = ctx.ref_audio.get("audios")
        if isinstance(entries, list):
            condition_audio_t.extend(
                int(entry["ref_audio_t"])
                for entry in entries
                if int(entry["ref_audio_t"]) > 0
            )
        elif ctx.ref_audio.get("ref_audio_t") is not None:
            ref_audio_t = int(ctx.ref_audio["ref_audio_t"])
            if ref_audio_t > 0:
                condition_audio_t.append(ref_audio_t)
    return condition_audio_t


def _apply_condition_noise_aug(
    ctx: _FullLoopContext,
    *,
    sampling: Any,
    imgvid_noise_aug: float,
    audio_noise_aug: float,
) -> None:
    """Apply condition noise augmentation to cond rows."""

    noise_visual_conditions = (
        ctx.cond_rows is not None and float(imgvid_noise_aug) < 1.0
    )
    noise_audio_conditions = (
        ctx.audio_ref_rows is not None and float(audio_noise_aug) < 1.0
    )
    if not (noise_visual_conditions or noise_audio_conditions):
        return

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )

    noise_seed = getattr(ctx.plan, "seed", None) if ctx.plan is not None else None
    if noise_seed is None:
        noise_seed = getattr(sampling, "seed", None)
    if noise_seed is None:
        noise_seed = 42

    if noise_visual_conditions:
        condition_shapes = _imgvid_condition_shapes(
            ref2va_blocks=ctx.ref2va_positive_blocks,
            keyframe=ctx.keyframe,
            is_ref2va=ctx.is_ref2va,
        )
        imgvid_cond_num_frames = len(condition_shapes)
        if not condition_shapes:
            raise ValueError("imgvid condition rows are missing shape metadata")
        # Imgvid conditions (ref2va blocks / keyframes) contribute one frame
        # count per condition entry.
        ctx.cond_rows = minimax_h3_imgvid_cond_noise_aug_rows(
            ctx.cond_rows,
            condition_shapes=condition_shapes,
            target_latent_t=ctx.latent_t,
            imgvid_cond_num_frames=imgvid_cond_num_frames,
            seed=int(noise_seed),
            noise_aug=float(imgvid_noise_aug),
        )

    if noise_audio_conditions:
        condition_audio_t = _condition_audio_lengths(ctx)
        if not condition_audio_t:
            raise ValueError("audio condition rows are missing length metadata")
        ctx.audio_ref_rows = minimax_h3_audio_cond_noise_aug_rows(
            ctx.audio_ref_rows,
            condition_audio_t=condition_audio_t,
            seed=int(noise_seed),
            noise_aug=float(audio_noise_aug),
        )


def _expand_initial_rows(
    ctx: _FullLoopContext,
    positive: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter target-row noise into the full packed layout when cond rows exist."""

    initial_video = ctx.state["initial_video_rows"]
    if ctx.include_cond:
        # layout target rows = noise; cond anchors appended by the loop
        n_video = positive.img_pos.shape[0]
        full = torch.zeros(int(n_video), initial_video.shape[1], dtype=torch.float32)
        full[positive.update_mask] = initial_video
        initial_video = full

    initial_audio = ctx.state["initial_audio_rows"]
    if ctx.audio_ref_rows is not None:
        n_audio = positive.audio_pos.shape[0]
        full_audio = torch.zeros(
            int(n_audio), initial_audio.shape[1], dtype=torch.float32
        )
        full_audio[positive.audio_update_mask] = initial_audio
        initial_audio = full_audio
    return initial_video, initial_audio


def _publish_full_loop_outputs(
    ctx: _FullLoopContext,
    *,
    batch: Req,
    positive: Any,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
) -> None:
    """Compose the generated target latents onto the batch."""

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_unpack_audio_tokens,
        minimax_h3_unpatchify_video_tokens,
    )

    target_rows = video_rows[positive.video_target_slice]
    # Keep latents on CUDA so decode can reuse them without a device round-trip;
    # decode autocast is enabled only for CUDA inputs.
    batch.latents = minimax_h3_unpatchify_video_tokens(
        target_rows,
        latent_shape=[ctx.latent_t, ctx.latent_h // 2, ctx.latent_w // 2, 24],
        patch_size=[1, 2, 2],
    )
    audio_target_rows = audio_rows[positive.audio_target_slice]
    batch.audio_latents = minimax_h3_unpack_audio_tokens(
        audio_target_rows, audio_t=ctx.audio_t * 2, audio_channel=2
    )
    if isinstance(ctx.state, dict):
        ctx.state["latent_h"] = ctx.latent_h
        ctx.state["latent_w"] = ctx.latent_w


__all__ = [
    "MiniMaxH3DenoisingStage",
    "minimax_h3_condition_noise_aug",
]
