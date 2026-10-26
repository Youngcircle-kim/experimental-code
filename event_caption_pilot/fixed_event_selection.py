"""Cross allocation and within-event selection on frozen event boundaries.

All scores and embeddings are supplied by the completed source experiment.
The algorithm never reads options, labels or prior QA outcomes. Embedding
distance is a redundancy proxy, not a measurement of answer evidence.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .data import array_digest
from .legacy_complexity_analysis import (
    _bounds,
    _choose,
    _vector,
    allocate_legacy,
    normalize_scores,
    representative_indices,
)
from .query_diversity_analysis import _unit_features

CONDITIONS = ("R_topk", "G_topk", "R_diverse", "G_diverse")
ALGORITHM_SPEC = {
    "version": "fixed-event-allocation-selection-factorial-v1",
    "source": "Frozen A_R_topk from the question-diversity experiment",
    "events": "Unchanged source events, candidate grid and question cosines",
    "R_allocation": "Exact source Legacy R allocation, reproduced and checked",
    "G_allocation": (
        "Anchor the highest question-cosine candidate in every event; spend "
        "the rest on highest-cosine unselected candidates globally; "
        "use resulting per-event counts as the fixed quota for both selectors"
    ),
    "topk": "Highest saved question cosine within each event's quota",
    "diverse": (
        "For quota k, restrict to the top min(2*k, capacity) by question "
        "cosine. Anchor the highest-cosine frame, then greedily choose the "
        "candidate with smallest maximum cosine to the selected event frames"
    ),
    "relevance_pool_ratio": 2,
    "diverse_scope": "Within each event, without cross-event suppression",
    "ties": (
        "Relevance ties: earlier index. Farthest-first ties: higher question "
        "cosine, then earlier index. Final QA frames are chronological"
    ),
    "features": "Preserve source bytes/hash; unit normalization in float64",
    "source_replay_tolerance": (
        "Only float64 quota probabilities/remainders allow absolute 1e-12 "
        "cross-platform exp roundoff; allocation and selected indices exact"
    ),
    "minimum_per_event": 1,
    "budget": "Exact unique frame budget; require budget >= event count",
    "diagnostic_scope": (
        "Cosine redundancy and temporal spread are proxies, not annotated "
        "evidence coverage. No accuracy-dependent parameter selection"
    ),
}


def _rank(indices, scores):
    indices = np.asarray(indices, dtype=np.int64)
    return indices[np.lexsort((indices, -scores[indices]))]


def _baseline(base, features, times, budget):
    if not isinstance(base, dict) or base.get("condition") != "A_R_topk":
        raise ValueError("Require the frozen A_R_topk source selection")
    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (int, np.integer))
        or budget < 1
    ):
        raise ValueError("Budget must be a positive integer")
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Candidate times must be finite and increasing")
    bounds = _bounds(base["events"], len(times))
    if not len(bounds) <= budget <= len(times):
        raise ValueError("Require event_count <= budget <= candidate_count")
    unit = _unit_features(features)
    if len(unit) != len(times):
        raise ValueError("Feature count differs from candidate timestamps")
    digest = array_digest(np.asarray(features))
    if base.get("feature_sha256") != digest:
        raise ValueError("Feature bytes, shape or dtype differ from source")
    scores = _vector(base["frame_scores_raw_cosine"], len(times), "cosine")
    if np.any(np.abs(scores) > 1 + 1e-6):
        raise ValueError("Question cosine lies outside [-1,1]")
    capacities = bounds[:, 1] - bounds[:, 0]
    representatives = representative_indices(bounds)
    event_max = np.asarray(
        [max(scores[ids]) for ids in representatives], dtype=np.float32
    )
    logits = event_max / np.float32(0.07)
    probabilities = np.exp(logits - logits.max())
    probabilities = (probabilities / probabilities.sum()).astype(np.float32)
    relevance = normalize_scores(probabilities)
    product = (np.ones(len(bounds), dtype=np.float32) + 0.01) * (
        relevance + 0.01
    )
    importance = normalize_scores(product)
    expected = {
        "capacities": capacities.tolist(),
        "representative_indices": representatives,
        "relevance_raw_cosine": event_max.tolist(),
        "relevance_softmax": probabilities.tolist(),
        "relevance_scores": relevance.tolist(),
        "base_importance_product": product.tolist(),
        "base_importance_scores": importance.tolist(),
        "importance_scores": importance.tolist(),
        **allocate_legacy(importance, capacities, int(budget)),
    }
    for key, value in expected.items():
        if base.get(key) != value:
            if key in {
                "weights_before_capacity_rounding",
                "raw_extra_quotas",
                "raw_remainders",
            }:
                actual = np.asarray(base.get(key), dtype=np.float64)
                if actual.shape == np.asarray(value).shape and np.allclose(
                    actual, value, rtol=0, atol=1e-12
                ):
                    continue
            raise ValueError(f"Frozen baseline {key} does not reproduce")
    selected = np.asarray(base["selected_indices"])
    topk = _choose(bounds, expected["allocation"], scores, True)
    if (
        selected.dtype.kind not in "iu"
        or selected.shape != (budget,)
        or selected.tolist() != topk
        or base.get("selected_timestamps") != times[topk].tolist()
    ):
        raise ValueError("Frozen baseline Top-K indices or times differ")
    return bounds, scores, unit, times, digest, expected


def _global_quota(bounds, scores, budget):
    anchors = [
        int(_rank(np.arange(start, stop), scores)[0]) for start, stop in bounds
    ]
    remaining = np.ones(len(scores), dtype=bool)
    remaining[anchors] = False
    additions = _rank(np.flatnonzero(remaining), scores)[
        : budget - len(anchors)
    ].tolist()
    indices = sorted(anchors + additions)
    event_ids = np.repeat(np.arange(len(bounds)), bounds[:, 1] - bounds[:, 0])
    quota = np.bincount(event_ids[indices], minlength=len(bounds)).tolist()
    if _choose(bounds, quota, scores, True) != indices:
        raise ValueError("Global quota does not reproduce constrained Top-K")
    return quota, anchors, additions


def _select_diverse(bounds, quota, scores, unit):
    result, details = [], []
    for (start, stop), count in zip(bounds, quota):
        pool = _rank(np.arange(start, stop), scores)[
            : min(2 * count, stop - start)
        ]
        chosen = [int(pool[0])]
        available = np.ones(len(pool), dtype=bool)
        available[0] = False
        maximum_similarity = np.clip(unit[pool] @ unit[pool[0]], -1.0, 1.0)
        while len(chosen) < count:
            positions = np.flatnonzero(available)
            order = np.lexsort(
                (
                    pool[positions],
                    -scores[pool[positions]],
                    maximum_similarity[positions],
                )
            )
            position = int(positions[order[0]])
            next_index = int(pool[position])
            chosen.append(next_index)
            available[position] = False
            maximum_similarity = np.maximum(
                maximum_similarity,
                np.clip(unit[pool] @ unit[next_index], -1.0, 1.0),
            )
        result.extend(chosen)
        details.append(
            {
                "pool_indices_by_relevance": pool.tolist(),
                "anchor_index": int(pool[0]),
                "greedy_selection_order": chosen,
            }
        )
    return sorted(result), details


def _diagnostics(indices, quota, scores, unit, times, baseline):
    budget = len(indices)
    selected = unit[indices]
    pairwise = np.clip(selected @ selected.T, -1.0, 1.0)
    pairs = pairwise[np.triu_indices(budget, 1)]
    selected_times = times[indices]
    span = float(times[-1] - times[0])
    if span:
        bins = np.minimum(
            ((selected_times - times[0]) / span * budget).astype(int),
            budget - 1,
        )
        span_fraction = float((selected_times[-1] - selected_times[0]) / span)
    else:
        bins = np.zeros(budget, dtype=int)
        span_fraction = 0.0
    positions = np.searchsorted(selected_times, times)
    left = selected_times[np.maximum(positions - 1, 0)]
    right = selected_times[np.minimum(positions, budget - 1)]
    nearest = np.minimum(np.abs(times - left), np.abs(times - right))
    weights = np.asarray(quota, dtype=np.float64) / budget
    return {
        "event_count": len(quota),
        "zero_frame_events": int(np.count_nonzero(np.asarray(quota) == 0)),
        "one_frame_events": int(np.count_nonzero(np.asarray(quota) == 1)),
        "minimum_exhausts_budget": budget == len(quota),
        "allocation_entropy": float(-np.sum(weights * np.log(weights))),
        "selected_mean_question_cosine": float(scores[indices].mean()),
        "selected_min_question_cosine": float(scores[indices].min()),
        "selected_feature_redundancy": float(pairs.mean())
        if len(pairs)
        else 0.0,
        "selected_temporal_span_fraction": span_fraction,
        "temporal_bin_count": budget,
        "occupied_temporal_bins": len(np.unique(bins)),
        "occupied_temporal_bin_fraction": len(np.unique(bins)) / budget,
        "mean_candidate_nearest_selected_seconds": float(nearest.mean()),
        "max_candidate_nearest_selected_seconds": float(nearest.max()),
        "changed_allocation_events_vs_R": sum(
            a != b for a, b in zip(quota, baseline["allocation"])
        ),
        "changed_frames_vs_R_topk": budget
        - len(set(indices) & set(baseline["selected_indices"])),
    }


def make_conditions(base_selection, features, times, budget=32):
    """Return four deterministic arms while preserving source R quotas.

    R/G quotas are each shared exactly by Top-K and diverse selection. G+Top-K
    maximizes the sum of saved question cosines subject to one frame per event.
    Diverse selection always preserves each event's most relevant frame and
    cannot select outside that event's fixed Top-2K relevance pool.
    """
    bounds, scores, unit, times, digest, baseline = _baseline(
        base_selection, features, times, budget
    )
    global_quota, anchors, additions = _global_quota(bounds, scores, budget)
    quotas = {"R": baseline["allocation"], "G": global_quota}
    selections = {}
    for name in CONDITIONS:
        allocation_name, selector = name.split("_")
        quota = quotas[allocation_name]
        topk = _choose(bounds, quota, scores, True)
        if selector == "topk":
            indices, details = topk, []
        else:
            indices, details = _select_diverse(bounds, quota, scores, unit)
        if len(indices) != budget or len(set(indices)) != budget:
            raise ValueError("Selection violates exact unique budget")
        diagnostics = _diagnostics(
            indices, quota, scores, unit, times, base_selection
        )
        diagnostics["changed_frames_vs_same_allocation_topk"] = budget - len(
            set(indices) & set(topk)
        )
        selections[name] = {
            "condition": name,
            "spec": deepcopy(ALGORITHM_SPEC),
            "source_condition": "A_R_topk",
            "events": bounds.tolist(),
            "capacities": baseline["capacities"],
            "frame_budget": int(budget),
            "frame_scores_raw_cosine": scores.tolist(),
            "feature_sha256": digest,
            "feature_shape": list(np.asarray(features).shape),
            "feature_dtype": np.asarray(features).dtype.str,
            "candidate_timestamps_sha256": array_digest(times),
            "allocation_policy": "legacy_R"
            if allocation_name == "R"
            else "minimum_one_global_question_topk",
            "allocation": list(quota),
            "minimum_per_event": 1,
            "within_event": "question_cosine_topk"
            if selector == "topk"
            else "top2k_relevance_gated_farthest_first",
            "relevance_pool_ratio": 2 if selector == "diverse" else None,
            "global_anchor_indices": anchors
            if allocation_name == "G"
            else None,
            "global_addition_indices_by_relevance": additions
            if allocation_name == "G"
            else None,
            "selection_steps": details,
            "selected_indices": indices,
            "selected_timestamps": times[indices].tolist(),
            "diagnostics": diagnostics,
        }
    return selections


def validate_selection(
    selection, name, base_selection, features, times, budget=32
):
    """Recompute source validation, quota, sequence and all diagnostics."""
    if name not in CONDITIONS:
        raise ValueError("Unknown fixed-event condition")
    expected = make_conditions(base_selection, features, times, budget)[name]
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"Fixed-event selection {key} differs")
