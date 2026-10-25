"""Legacy pixel-complexity allocation on frozen candidate partitions.

This ports scoring/allocation from Legacy 7a711d2, not its decoder, detector,
CLIP model or QA model. Callers supply frozen boundaries, candidate cosines and
directly resized 64-pixel RGB representatives. No model imports are needed.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import numpy as np

CONDITIONS = (
    "legacy_R_uniform",
    "legacy_R_topk",
    "legacy_CR_uniform",
    "legacy_CR_topk",
)
ALGORITHM_SPEC = {
    "version": "legacy-7a711d2-pixel-complexity-frozen-partitions-v1",
    "reference_commit": "7a711d2",
    "scope": (
        "Legacy scoring and allocation on existing candidate partitions and "
        "cached SigLIP2 cosine scores; not end-to-end Legacy reproduction"
    ),
    "representatives": (
        "min(4, capacity); numpy.linspace(start, stop-1, n, dtype=int)"
    ),
    "pixels": "original decoded RGB resized with PIL to 64x64, float32 / 255",
    "motion": "float(mean(mean(abs(frames[1:]-frames[:-1]), axis=(1,2,3))))",
    "variance": "float(var(frames)) over representative, spatial and RGB axes",
    "complexity": "float32(motion+variance); singleton raw=1e-6; minmax twice",
    "minmax": "float32; range<1e-6 -> ones; otherwise (x-min)/(max-min+1e-6)",
    "relevance": (
        "representative maximum cosine; float32 softmax at 0.07; minmax"
    ),
    "importance": "minmax((complexity+0.01)*(relevance+0.01))",
    "control": (
        "R replaces twice-normalized complexity with ones before product"
    ),
    "allocation": (
        "Legacy softmax(max(importance,1e-6)/0.3); feasible minimum one; "
        "floor quotas, capacity overflow by importance, then largest remainder"
    ),
    "capacity_corner": (
        "Require budget within total capacity; reset exhausted remainders "
        "only among events with headroom, preventing the Legacy "
        "over-capacity corner"
    ),
    "uniform": (
        "Legacy numpy.linspace including endpoints; one frame uses start"
    ),
    "topk": (
        "all event candidates ranked by cached question cosine; earlier ties"
    ),
    "precision": (
        "Scoring float32, allocation float64; numpy float32 softmax replays "
        "cached cosine values, not the original torch/CLIP dot products"
    ),
}


def _bounds(bounds, count=None):
    values = np.asarray(bounds)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not len(values)
        or values.dtype.kind not in "iu"
        or values[0, 0] != 0
        or np.any(values[:, 0] >= values[:, 1])
        or np.any(values[1:, 0] != values[:-1, 1])
        or (count is not None and values[-1, 1] != count)
    ):
        raise ValueError("Events must partition all candidates exactly")
    return values.astype(np.int64, copy=False)


def _vector(value, length, name, dtype=np.float64):
    result = np.asarray(value, dtype=dtype)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name} vector")
    return result


def normalize_scores(values):
    """Legacy float32 min-max, including its constant-vector -> ones rule."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite vector")
    if not len(values):
        return values
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-6:
        return np.ones_like(values, dtype=np.float32)
    return ((values - low) / (high - low + 1e-6)).astype(np.float32)


def representative_indices(bounds):
    """Return up to four Legacy representatives for every half-open event."""
    return [
        np.linspace(start, stop - 1, min(4, stop - start), dtype=int).tolist()
        for start, stop in _bounds(bounds)
    ]


def pixel_complexity(bounds, scoring_frames):
    """Compute Legacy pixel components once per frozen partition.

    scoring_frames maps candidate index to a PIL-resized 64x64 RGB float32
    array in [0,1]. The caller handles decoding and resizing explicitly.
    """
    bounds = _bounds(bounds)
    representatives = representative_indices(bounds)
    motion, variance, raw = [], [], []
    digest = sha256()
    for index in sorted({i for indices in representatives for i in indices}):
        if index not in scoring_frames:
            raise ValueError(f"Missing scoring representative: {index}")
        frame = np.asarray(scoring_frames[index])
        if (
            frame.shape != (64, 64, 3)
            or frame.dtype != np.float32
            or not np.isfinite(frame).all()
            or np.any(frame < 0)
            or np.any(frame > 1)
        ):
            raise ValueError(
                "Scoring frames must be finite 64x64 RGB float32 [0,1]"
            )
        digest.update(str((index, frame.shape, frame.dtype.str)).encode())
        digest.update(np.ascontiguousarray(frame).tobytes())
    for indices in representatives:
        if len(indices) == 1:
            motion.append(0.0)
            variance.append(0.0)
            raw.append(1e-6)
            continue
        frames = np.stack([scoring_frames[i] for i in indices])
        change = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2, 3))
        motion_value, variance_value = (
            float(np.mean(change)),
            float(np.var(frames)),
        )
        motion.append(motion_value)
        variance.append(variance_value)
        raw.append(motion_value + variance_value)
    raw = np.asarray(raw, dtype=np.float32)
    first = normalize_scores(raw)
    return {
        "spec": deepcopy(ALGORITHM_SPEC),
        "events": bounds.tolist(),
        "representative_indices": representatives,
        "source_pixels_sha256": digest.hexdigest(),
        "motion": motion,
        "variance": variance,
        "raw": raw.tolist(),
        "normalized_once": first.tolist(),
        "normalized": normalize_scores(first).tolist(),
    }


def _validate_complexity(record, bounds):
    if (
        record.get("spec") != ALGORITHM_SPEC
        or record.get("events") != bounds.tolist()
        or record.get("representative_indices")
        != representative_indices(bounds)
    ):
        raise ValueError("Complexity recipe or event representatives changed")
    digest = record.get("source_pixels_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid scoring-pixel digest")
    n = len(bounds)
    motion = _vector(record["motion"], n, "motion")
    variance = _vector(record["variance"], n, "variance")
    if (
        np.any(motion < 0)
        or np.any(motion > 1)
        or np.any(variance < 0)
        or np.any(variance > 0.250001)
    ):
        raise ValueError(
            "Pixel complexity components exceed [0,1] pixel bounds"
        )
    raw = (motion + variance).astype(np.float32)
    singles = bounds[:, 1] - bounds[:, 0] == 1
    if np.any(motion[singles] != 0) or np.any(variance[singles] != 0):
        raise ValueError("Singleton complexity components must be zero")
    raw[singles] = np.float32(1e-6)
    first = normalize_scores(raw)
    final = normalize_scores(first)
    for name, expected in (
        ("raw", raw),
        ("normalized_once", first),
        ("normalized", final),
    ):
        if not np.array_equal(
            _vector(record[name], n, name), expected.astype(float)
        ):
            raise ValueError(f"Saved complexity {name} does not reproduce")
    return final


def allocate_legacy(importance, capacities, budget=32):
    """Legacy quota calculation with strict capacity and exact-budget checks.

    The exhausted-remainder corner restarts the ranking within headroom.
    """
    capacities = np.asarray(capacities)
    if (
        capacities.ndim != 1
        or not len(capacities)
        or capacities.dtype.kind not in "iu"
        or np.any(capacities < 1)
        or isinstance(budget, bool)
        or not isinstance(budget, (int, np.integer))
        or not 0 <= budget <= sum(int(n) for n in capacities)
    ):
        raise ValueError(
            "Require positive capacities and feasible integer budget"
        )
    scores = np.maximum(
        _vector(importance, len(capacities), "importance"), 1e-6
    )
    logits = scores / 0.3
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= np.sum(probabilities)
    minimum = int(budget >= len(capacities))
    allocation = np.full(len(capacities), minimum, dtype=np.int64)
    remaining = int(budget - allocation.sum())
    raw = probabilities * remaining
    floor = np.floor(raw).astype(np.int64)
    allocation += floor
    remainders = raw - floor
    overflow = int(np.maximum(allocation - capacities, 0).sum())
    allocation = np.minimum(allocation, capacities)
    priority = np.argsort(-scores)
    while overflow:
        for index in priority:
            if allocation[index] < capacities[index]:
                allocation[index] += 1
                overflow -= 1
                if not overflow:
                    break
    remaining_remainders = remainders.copy()
    resets = 0
    while int(allocation.sum()) < budget:
        headroom = allocation < capacities
        masked = np.where(headroom, remaining_remainders, -np.inf)
        if not np.isfinite(masked).any():
            # Legacy reused argmax(-inf) and could exceed a full event's cap.
            # Restart only the residual ranking among still-available events.
            remaining_remainders = remainders.copy()
            masked = np.where(headroom, remaining_remainders, -np.inf)
            resets += 1
        index = int(np.argmax(masked))
        allocation[index] += 1
        remaining_remainders[index] = -np.inf
    if int(allocation.sum()) != budget or np.any(allocation > capacities):
        raise ValueError("Legacy allocation failed exact capacity constraints")
    return {
        "allocation": allocation.tolist(),
        "allocation_scores": scores.tolist(),
        "weights_before_capacity_rounding": probabilities.tolist(),
        "minimum_per_event": minimum,
        "residual_budget_after_minimum": remaining,
        "raw_extra_quotas": raw.tolist(),
        "raw_remainders": remainders.tolist(),
        "capacity_overflow_redistributed": int(
            np.maximum(minimum + floor - capacities, 0).sum()
        ),
        "capacity_remainder_resets": resets,
        "temperature": 0.3,
    }


def _choose(bounds, allocation, frame_scores, topk):
    selected = []
    for (start, stop), number in zip(bounds, allocation):
        if not number:
            continue
        if topk:
            candidates = np.arange(start, stop)
            order = np.lexsort((candidates, -frame_scores[candidates]))
            selected.extend(candidates[order[:number]].tolist())
        else:
            selected.extend(
                np.linspace(start, stop - 1, number, dtype=int).tolist()
            )
    return sorted(selected)


def make_conditions(bounds, frame_scores, times, complexity, budget=32):
    """Cross the Legacy complexity contribution with uniform versus top-k."""
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Require finite strictly increasing candidate timestamps"
        )
    bounds = _bounds(bounds, len(times))
    frame_scores = _vector(frame_scores, len(times), "candidate cosine")
    if np.any(np.abs(frame_scores) > 1 + 1e-6):
        raise ValueError("Candidate cosine outside [-1,1]")
    normalized_complexity = _validate_complexity(complexity, bounds)
    representatives = representative_indices(bounds)
    event_max = np.asarray(
        [max(frame_scores[ids]) for ids in representatives], dtype=np.float32
    )
    logits = event_max / np.float32(0.07)
    softmax = np.exp(logits - np.max(logits))
    softmax = (softmax / np.sum(softmax)).astype(np.float32)
    relevance = normalize_scores(softmax)
    capacities = bounds[:, 1] - bounds[:, 0]
    selections = {}
    for name in CONDITIONS:
        use_complexity = name.startswith("legacy_CR_")
        topk = name.endswith("_topk")
        component = (
            normalized_complexity
            if use_complexity
            else np.ones(len(bounds), dtype=np.float32)
        )
        product = (component + 0.01) * (relevance + 0.01)
        importance = normalize_scores(product)
        quota = allocate_legacy(importance, capacities, budget)
        indices = _choose(bounds, quota["allocation"], frame_scores, topk)
        if len(indices) != budget or len(set(indices)) != budget:
            raise ValueError(
                "Selection does not contain the exact unique budget"
            )
        allocation = np.asarray(quota["allocation"])
        weights = np.asarray(quota["weights_before_capacity_rounding"])
        selections[name] = {
            "condition": name,
            "spec": deepcopy(ALGORITHM_SPEC),
            "events": bounds.tolist(),
            "capacities": capacities.tolist(),
            "representative_indices": representatives,
            "complexity": deepcopy(complexity),
            "complexity_enabled": use_complexity,
            "complexity_component": component.tolist(),
            "frame_scores_raw_cosine": frame_scores.tolist(),
            "relevance_raw_cosine": event_max.tolist(),
            "relevance_softmax": softmax.tolist(),
            "relevance_scores": relevance.tolist(),
            "importance_product": product.tolist(),
            "importance_scores": importance.tolist(),
            "allocation_policy": "legacy_relevance_complexity"
            if use_complexity
            else "legacy_relevance_only",
            "within_event": "question_cosine_topk"
            if topk
            else "legacy_uniform_first_if_one",
            **quota,
            "selected_indices": indices,
            "selected_timestamps": times[indices].tolist(),
            "diagnostics": {
                "event_count": len(bounds),
                "zero_frame_events": int((allocation == 0).sum()),
                "one_frame_events": int((allocation == 1).sum()),
                "multi_frame_events": int((allocation >= 2).sum()),
                "allocation_entropy": float(
                    -np.sum(weights * np.log(weights))
                ),
                "minimum_exhausts_budget": quota[
                    "residual_budget_after_minimum"
                ]
                == 0,
                "selected_mean_question_cosine": float(
                    frame_scores[indices].mean()
                )
                if indices
                else None,
            },
        }
    for name, selection in selections.items():
        reference = selections[
            "legacy_R_" + ("topk" if name.endswith("_topk") else "uniform")
        ]
        selection["diagnostics"].update(
            {
                "changed_allocation_events_vs_R": sum(
                    a != b
                    for a, b in zip(
                        selection["allocation"], reference["allocation"]
                    )
                ),
                "changed_frames_vs_R": budget
                - len(
                    set(selection["selected_indices"])
                    & set(reference["selected_indices"])
                ),
            }
        )
    return selections


def validate_selection(selection, name, times, budget=32):
    """Recompute saved sufficient statistics, allocation and exact indices."""
    if name not in CONDITIONS:
        raise ValueError("Unknown Legacy condition")
    expected = make_conditions(
        selection["events"],
        selection["frame_scores_raw_cosine"],
        times,
        selection["complexity"],
        budget,
    )[name]
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"Legacy selection {key} does not reproduce")
