"""Question-conditioned semantic diversity on a frozen Legacy R policy.

The model-free A/B/C comparison changes only event allocation importance.
It reuses saved per-candidate question cosines for relevance and within-event
Top-K. Diversity is an embedding-space proxy, not a measure of answer evidence.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np

from .data import array_digest
from .legacy_complexity_analysis import (
    _bounds,
    _choose,
    _vector,
    allocate_legacy,
)
from .legacy_complexity_analysis import (
    validate_selection as validate_legacy_selection,
)

CONDITIONS = ("A_R_topk", "B_R_Cvis_topk", "C_R_Cq_topk")
ALGORITHM_SPEC = {
    "version": "question-weighted-event-embedding-diversity-v1",
    "scope": (
        "A/B/C allocation ablation on the frozen Legacy relevance-only Top-K "
        "selection; semantic dispersion is a proxy, not answer evidence"
    ),
    "baseline": (
        "Exact legacy_R_topk: representative maximum cosine from up to four "
        "uniform representatives; relevance softmax 0.07 and Legacy minmax"
    ),
    "features": (
        "All event candidate SigLIP2 embeddings, preserving frozen source "
        "feature dtype; unit normalization and statistics in float64"
    ),
    "Cvis": "1 - squared_norm(mean(unit_frame_embeddings))",
    "Cq": (
        "1 - squared_norm(sum(softmax(saved_frame_cosines / 0.07) * "
        "unit_frame_embeddings)); softmax over each event's candidates"
    ),
    "diversity_bounds": (
        "[0,1]; singleton zero; clamp only numerical roundoff up to 1e-12; "
        "no event-wise minmax normalization"
    ),
    "strength": 0.5,
    "question_weight_temperature": 0.07,
    "importance": (
        "A: exact frozen R importance; B: R importance * (1+0.5*Cvis); "
        "C: R importance * (1+0.5*Cq); no normalization after multiplication"
    ),
    "allocation": (
        "Unchanged Legacy allocator: softmax(max(importance,1e-6)/0.3), "
        "feasible minimum one, capacity handling and exact budget 32"
    ),
    "within_event": (
        "Unchanged Top-K by saved per-candidate question cosine; earlier "
        "candidate index wins ties; final indices chronological"
    ),
    "zero_strength": "Exact bypass to frozen allocation and selected frames",
}


def feature_sha256(features):
    """Bind cache bytes, shape and dtype before float64 normalization."""
    _unit_features(features)
    return array_digest(np.asarray(features))


def _unit_features(features):
    values = np.asarray(features)
    if (
        values.ndim != 2
        or not all(values.shape)
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
    ):
        raise ValueError("Features must be a finite nonempty floating matrix")
    result = values.astype(np.float64, copy=True)
    # Scaling first avoids overflow/underflow for finite nonzero embeddings.
    scale = np.max(np.abs(result), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Feature rows must have nonzero norm")
    result /= scale
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def _parameter(value, name, *, positive):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"Invalid {name}")
    return float(value)


def _dispersion(mean):
    value = float(1.0 - np.dot(mean, mean))
    if not np.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
        raise ValueError("Normalized embedding diversity exceeds [0,1]")
    return min(1.0, max(0.0, value))


def event_diversity(features, frame_scores, bounds, temperature=0.07):
    """Compute unweighted and question-weighted diversity on every candidate.

    Frame scores must be saved question cosines, not newly encoded scores.
    Event-local stable softmax preserves a constant-relevance control.
    """
    temperature = _parameter(temperature, "temperature", positive=True)
    unit = _unit_features(features)
    bounds = _bounds(bounds, len(unit))
    scores = _vector(frame_scores, len(unit), "candidate cosine")
    if np.any(np.abs(scores) > 1 + 1e-6):
        raise ValueError("Candidate cosine outside [-1,1]")
    visual, question, effective, maximum = [], [], [], []
    for start, stop in bounds:
        event = unit[start:stop]
        # Subtract before division so low temperatures do not overflow.
        with np.errstate(over="ignore"):
            logits = (scores[start:stop] - scores[start:stop].max()) / (
                temperature
            )
        weights = np.exp(logits)
        weights /= weights.sum()
        if len(event) == 1:
            visual.append(0.0)
            question.append(0.0)
        else:
            visual.append(_dispersion(event.mean(axis=0)))
            question.append(_dispersion(weights @ event))
        effective.append(float(1.0 / np.dot(weights, weights)))
        maximum.append(float(weights.max()))
    return {
        "events": bounds.tolist(),
        "Cvis": visual,
        "Cq": question,
        "question_weight_effective_candidates": effective,
        "question_weight_maximum": maximum,
        "temperature": temperature,
        "feature_sha256": array_digest(np.asarray(features)),
        "feature_shape": list(unit.shape),
        "feature_dtype": np.asarray(features).dtype.str,
    }


def make_conditions(
    base_selection, features, times, strength=0.5, temperature=0.07
):
    """Return A/B/C while preserving all frozen baseline inference choices.

    ``base_selection`` is the complete saved ``legacy_R_topk`` selection,
    optionally including QA result fields. Those fields are never propagated.
    """
    strength = _parameter(strength, "strength", positive=False)
    temperature = _parameter(temperature, "temperature", positive=True)
    if (
        not isinstance(base_selection, dict)
        or base_selection.get("condition") != "legacy_R_topk"
    ):
        raise ValueError("Require the frozen legacy_R_topk base selection")
    validate_legacy_selection(base_selection, "legacy_R_topk", times, 32)
    times = np.asarray(times, dtype=np.float64)
    bounds = _bounds(base_selection["events"], len(times))
    if np.asarray(features).shape[0:1] != (len(times),):
        raise ValueError("Feature count differs from candidate timestamps")
    diversity = event_diversity(
        features,
        base_selection["frame_scores_raw_cosine"],
        bounds,
        temperature,
    )
    unit_features = _unit_features(features)
    capacities = bounds[:, 1] - bounds[:, 0]
    baseline = _vector(
        base_selection["importance_scores"], len(bounds), "base importance"
    )
    frame_scores = np.asarray(base_selection["frame_scores_raw_cosine"])
    base_quota = allocate_legacy(baseline, capacities, 32)
    # Every base quota field must reproduce, including diagnostic metadata.
    for key, value in base_quota.items():
        if base_selection.get(key) != value:
            raise ValueError(f"Frozen baseline quota {key} differs")
    records = {}
    for name, metric in zip(CONDITIONS, (None, "Cvis", "Cq")):
        component = (
            np.zeros(len(bounds), dtype=np.float64)
            if metric is None
            else np.asarray(diversity[metric], dtype=np.float64)
        )
        bypass = metric is None or strength == 0
        importance = (
            baseline.copy()
            if bypass
            else baseline * (1.0 + strength * component)
        )
        if not np.isfinite(importance).all():
            raise ValueError("Diversity strength overflows importance")
        quota = (
            deepcopy(base_quota)
            if bypass
            else allocate_legacy(importance, capacities, 32)
        )
        indices = (
            deepcopy(base_selection["selected_indices"])
            if bypass
            else _choose(bounds, quota["allocation"], frame_scores, True)
        )
        if len(indices) != 32 or len(set(indices)) != 32:
            raise ValueError("Require exactly 32 distinct selected frames")
        allocation = np.asarray(quota["allocation"])
        weights = np.asarray(quota["weights_before_capacity_rounding"])
        positive_weights = weights[weights > 0]
        selected_features = unit_features[indices]
        selected_cosines = selected_features @ selected_features.T
        redundancy = float((selected_cosines.sum() - 32) / (32 * 31))
        records[name] = {
            "condition": name,
            "spec": deepcopy(ALGORITHM_SPEC),
            "source_condition": "legacy_R_topk",
            "source_spec": deepcopy(base_selection["spec"]),
            "events": bounds.tolist(),
            "capacities": capacities.tolist(),
            "representative_indices": deepcopy(
                base_selection["representative_indices"]
            ),
            **{
                key: deepcopy(base_selection[key])
                for key in (
                    "frame_scores_raw_cosine",
                    "relevance_raw_cosine",
                    "relevance_softmax",
                    "relevance_scores",
                )
            },
            "base_importance_scores": baseline.tolist(),
            "base_importance_product": deepcopy(
                base_selection["importance_product"]
            ),
            "importance_scores": importance.tolist(),
            "complexity": deepcopy(diversity),
            "complexity_metric": metric,
            "complexity_component": component.tolist(),
            "complexity_enabled": not bypass,
            "strength": strength,
            "question_weight_temperature": temperature,
            "feature_sha256": diversity["feature_sha256"],
            "allocation_policy": "legacy_R_times_bounded_semantic_diversity",
            "within_event": "question_cosine_topk",
            **quota,
            "selected_indices": indices,
            "selected_timestamps": times[indices].tolist(),
            "diagnostics": {
                "event_count": len(bounds),
                "zero_frame_events": int((allocation == 0).sum()),
                "one_frame_events": int((allocation == 1).sum()),
                "multi_frame_events": int((allocation >= 2).sum()),
                "allocation_entropy": float(
                    -np.sum(positive_weights * np.log(positive_weights))
                ),
                "minimum_exhausts_budget": quota[
                    "residual_budget_after_minimum"
                ]
                == 0,
                "selected_mean_question_cosine": float(
                    frame_scores[indices].mean()
                ),
                "selected_feature_redundancy": redundancy,
                "changed_allocation_events_vs_A": int(
                    np.sum(allocation != base_selection["allocation"])
                ),
                "changed_frames_vs_A": 32
                - len(set(indices) & set(base_selection["selected_indices"])),
            },
        }
    return records


def validate_selection(
    selection,
    name,
    base_selection,
    features,
    times,
    strength=0.5,
    temperature=0.07,
):
    """Recompute every saved sufficient statistic, allocation and index."""
    if name not in CONDITIONS:
        raise ValueError("Unknown query-diversity condition")
    expected = make_conditions(
        base_selection, features, times, strength, temperature
    )[name]
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"Query-diversity selection {key} differs")
