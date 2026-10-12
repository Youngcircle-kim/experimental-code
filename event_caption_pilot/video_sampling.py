"""Shared, budget-independent candidate plans using real native frames only."""

from __future__ import annotations

import math

SAMPLING_POLICY = "base_fps_plus_uniform_native_grid_quantiles_v1"


def checked_native_frame_count(value, fps, minimum):
    """Require explicit finite FPS and an integer container frame count."""
    detail = (
        f"source_fps={fps!r}, declared_native_frames={value!r}, "
        f"required_minimum={minimum}, observed_candidates=not_decoded"
    )
    try:
        valid_fps = (
            not isinstance(fps, bool) and math.isfinite(fps) and fps > 0
        )
        count = float(value)
        valid_count = (
            not isinstance(value, bool)
            and math.isfinite(count)
            and count > 0
            and count.is_integer()
        )
    except (TypeError, ValueError, OverflowError):
        valid_fps, valid_count = False, False
    if not valid_fps or not valid_count:
        raise ValueError(
            "Adaptive candidate sampling requires positive finite source FPS "
            "and a positive integer declared native frame count; " + detail
        )
    return int(count)


def _base_native_indices(config, fps, native_count):
    """Reproduce the streaming threshold test without multiplication rounding.

    Binary search finds the first native index satisfying the original
    index / fps >= next_observation comparison. The threshold still advances
    by exactly the original floating-point addition after each observation.
    """
    step = 1.0 / min(config.candidate_fps, fps)
    next_observation, start, indices = 0.0, 0, []
    while start < native_count:
        if (native_count - 1) / fps < next_observation:
            break
        low, high = start, native_count - 1
        while low < high:
            middle = (low + high) // 2
            if middle / fps >= next_observation:
                high = middle
            else:
                low = middle + 1
        indices.append(low)
        if len(indices) > config.max_candidate_frames:
            raise ValueError(
                "Base candidate plan exceeds max_candidate_frames: "
                f"observed_candidates={len(indices)}, source_fps={fps}, "
                f"declared_native_frames={native_count}, "
                "required_minimum="
                f"{getattr(config, 'min_candidate_frames', 0)}, "
                f"maximum_candidates={config.max_candidate_frames}"
            )
        next_observation += step
        start = low + 1
    return indices


def plan_native_candidates(config, fps, native_count):
    """Return chronological native indices and an auditable sampling record.

    Keep every original FPS-based candidate. If fewer than the configured
    minimum exist, add evenly spread quantiles from a full-video native grid.
    Grid and quantile arithmetic is integer-only; no frames are synthesized,
    duplicated or selected using a question, answer or QA frame budget.
    """
    minimum = getattr(config, "min_candidate_frames", 0)
    if (
        type(minimum) is not int
        or minimum < 0
        or minimum > config.max_candidate_frames
        or not math.isfinite(config.candidate_fps)
        or config.candidate_fps <= 0
    ):
        raise ValueError("Invalid candidate minimum, maximum or sampling FPS")
    native_count = checked_native_frame_count(native_count, fps, minimum)
    original = _base_native_indices(config, fps, native_count)
    if native_count < minimum:
        raise ValueError(
            "Declared native frame count cannot support candidate minimum: "
            f"planned_base_candidates={len(original)}, source_fps={fps}, "
            f"declared_native_frames={native_count}, "
            f"required_minimum={minimum}, observed_candidates=not_decoded"
        )
    selected = set(original)
    needed = max(0, minimum - len(selected))
    if needed:
        # minimum >= 2 here: every positive-count source contributes index 0.
        grid = [
            i * (native_count - 1) // (minimum - 1) for i in range(minimum)
        ]
        available = [index for index in grid if index not in selected]
        additions = [
            available[(2 * j + 1) * len(available) // (2 * needed)]
            for j in range(needed)
        ]
        selected.update(additions)
    indices = tuple(sorted(selected))
    if len(indices) != max(len(original), minimum):
        raise ValueError("Native candidate plan did not satisfy exact minimum")
    audit = {
        "policy": SAMPLING_POLICY,
        "min_candidate_frames": minimum,
        "source_fps": fps,
        "native_frame_count": native_count,
        "base_candidate_count": len(original),
        "candidate_count": len(indices),
        "densified": bool(needed),
    }
    return indices, audit


def native_candidate_indices(config, fps, native_count):
    """Expose the deterministic native-index plan without its audit record."""
    return plan_native_candidates(config, fps, native_count)[0]


def check_decoded_native_count(
    config, fps, declared, decoded, observed_candidates
):
    """Reject count disagreement after a complete sequential read attempt."""
    minimum = getattr(config, "min_candidate_frames", 0)
    if decoded != declared or observed_candidates < minimum:
        raise ValueError(
            "Native frame count/candidate minimum validation failed: "
            f"decoded_native_frames={decoded}, "
            f"declared_native_frames={declared}, "
            f"source_fps={fps}, observed_candidates={observed_candidates}, "
            f"required_minimum={minimum}"
        )
