"""Frozen B32 D2 allocation-by-within-event ablations and paired analysis."""

from __future__ import annotations

import numpy as np

from .algorithms import (
    allocate_frames,
    event_features,
    normalize_rows,
    sample_indices,
)
from .paired_reanalysis import clustered_means
from .types import Event

CONDITIONS = ("D_topk", "B", "B_topk")
BASELINES = ("uniform", "frame_top", "temporal_bin", "C", "D")
ARMS = (*BASELINES, *CONDITIONS)
COMPARISONS = {
    "D_topk_minus_D": ("D_topk", "D"),
    "B_minus_D": ("B", "D"),
    "B_topk_minus_B": ("B_topk", "B"),
    "B_topk_minus_D_topk": ("B_topk", "D_topk"),
    "B_topk_minus_D": ("B_topk", "D"),
    **{
        f"{arm}_minus_{baseline}": (arm, baseline)
        for arm in CONDITIONS
        for baseline in ("uniform", "temporal_bin", "C")
    },
}
FRAME_BUDGET = 32
INTERACTION = "interaction"


def _integers(values, name):
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain one-dimensional integers")
    return result


def _close(actual, expected, name):
    actual, expected = (
        np.asarray(value, dtype=float) for value in (actual, expected)
    )
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or not np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    ):
        raise ValueError(f"Original D {name} does not reproduce")


def make_conditions(features, query, times, d_plan, config):
    """Freeze D2 events; vary only allocation policy and within-event choice.

    Inputs contain no options, gold answers, captions or evidence labels.
    D_topk retains the exact saved D allocation. B and B_topk share a single
    candidate-count proportional allocation, with no minimum event quota.
    """
    config.validate()
    if config.frame_budget != FRAME_BUDGET:
        raise ValueError("Long ablations require exactly 32 frames")
    features = normalize_rows(features, config.normalization_epsilon)
    query = normalize_rows(
        np.asarray(query)[None, :], config.normalization_epsilon
    )[0]
    times = np.asarray(times, dtype=float)
    count = len(features)
    if (
        count < FRAME_BUDGET
        or query.shape != (features.shape[1],)
        or times.shape != (count,)
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Invalid candidate features, query or timestamps")
    if (
        d_plan.get("detector") != "D2"
        or d_plan.get("allocation_policy") != "question_relevance"
        or d_plan.get("temperature") != config.allocation_temperature
    ):
        raise ValueError("Require the frozen D2 question-relevance plan")
    bounds = np.asarray(d_plan["events"])
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or not len(bounds)
        or bounds.dtype.kind not in {"i", "u"}
        or bounds[0, 0] != 0
        or bounds[-1, 1] != count
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(bounds[1:, 0] != bounds[:-1, 1])
    ):
        raise ValueError("D events must partition all candidates exactly")
    events = [Event(int(a), int(b)) for a, b in bounds]
    capacities = bounds[:, 1] - bounds[:, 0]
    if not np.array_equal(
        _integers(d_plan["capacities"], "D capacities"), capacities
    ):
        raise ValueError("D capacities disagree with its frozen events")
    # portable_budget.make_selection normalizes once for frame controls;
    # event_factorial.select_plans normalizes again before event_features.
    # Reproduce both passes for exact saved integer-allocation checks, while
    # retaining the original frame-control cosine for within-event ranking.
    event_input = normalize_rows(features, config.normalization_epsilon)
    event_query = normalize_rows(
        query[None, :], config.normalization_epsilon
    )[0]
    relevance = (
        event_features(event_input, events, config.normalization_epsilon)
        @ event_query
    )
    _close(relevance, d_plan["relevance_raw_cosine"], "relevance")
    _close(relevance, d_plan["allocation_scores"], "allocation scores")
    original = _integers(d_plan["allocation"], "D allocation")
    reconstructed = allocate_frames(
        relevance, capacities, FRAME_BUDGET, config.allocation_temperature
    )
    if not np.array_equal(original, reconstructed):
        raise ValueError("Original D allocation does not reproduce")
    original_indices = [
        int(i)
        for event, number in zip(events, original)
        for i in sample_indices(event.start, event.stop, int(number))
    ]
    if not np.array_equal(
        _integers(d_plan["selected_indices"], "D selection"), original_indices
    ):
        raise ValueError("Original D uniform selection does not reproduce")
    if not np.array_equal(
        np.asarray(d_plan["selected_timestamps"], dtype=float),
        times[original_indices],
    ):
        raise ValueError("Original D selected timestamps do not reproduce")
    proportional = allocate_frames(
        np.log(capacities), capacities, FRAME_BUDGET, 1.0
    )
    frame_scores = features @ query
    bins = np.empty(count, dtype=int)
    for index, part in enumerate(np.array_split(np.arange(count), 32)):
        bins[part] = index
    plans = {}
    for arm in CONDITIONS:
        allocation = original if arm == "D_topk" else proportional
        selected = []
        for event, number in zip(events, allocation):
            if arm == "B":
                chosen = sample_indices(event.start, event.stop, int(number))
            else:
                candidates = np.arange(event.start, event.stop)
                ranking = np.lexsort((candidates, -frame_scores[candidates]))
                chosen = candidates[ranking[: int(number)]]
            selected.extend(int(i) for i in chosen)
        selected.sort()
        if len(selected) != 32 or len(set(selected)) != 32:
            raise ValueError("New condition violates the exact unique budget")
        scores = relevance if arm == "D_topk" else np.log(capacities)
        temperature = config.allocation_temperature if arm == "D_topk" else 1
        weights = np.exp((scores - scores.max()) / temperature)
        weights /= weights.sum()
        positive = weights[weights > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        overlap = len(set(selected) & set(original_indices))
        visual = features[selected]
        plans[arm] = {
            "detector": "D2",
            "allocation_policy": (
                "question_relevance" if arm == "D_topk" else "candidate_count"
            ),
            "within_event": (
                "uniform" if arm == "B" else "question_cosine_topk"
            ),
            "events": bounds.tolist(),
            "capacities": capacities.tolist(),
            "relevance_raw_cosine": relevance.tolist(),
            "allocation_scores": scores.tolist(),
            "temperature": float(temperature),
            "weights_before_capacity_rounding": weights.tolist(),
            "allocation": allocation.tolist(),
            "selected_indices": selected,
            "selected_timestamps": times[selected].tolist(),
            "diagnostics": {
                "n_events": len(events),
                "events_exceed_budget": len(events) > FRAME_BUDGET,
                "zero_allocation_events": int((allocation == 0).sum()),
                "zero_allocation_fraction": float(np.mean(allocation == 0)),
                "event_coverage_fraction": float(np.mean(allocation > 0)),
                "unallocated_candidate_fraction": float(
                    capacities[allocation == 0].sum() / count
                ),
                "multi_frame_events": int((allocation > 1).sum()),
                "max_event_allocation": int(allocation.max()),
                "weight_entropy": entropy,
                "normalized_weight_entropy": (
                    entropy / float(np.log(len(events)))
                    if len(events) > 1 else 0.0
                ),
                "effective_weighted_events": float(np.exp(entropy)),
                "fixed_bin_coverage": len(set(bins[selected])) / 32,
                "max_candidate_time_gap_seconds": float(
                    np.diff(np.r_[times[0], times[selected], times[-1]]).max()
                ),
                "selected_mean_question_cosine": float(
                    frame_scores[selected].mean()
                ),
                "selected_feature_redundancy": float(
                    ((visual @ visual.T).sum() - 32) / (32 * 31)
                ),
                "overlap_with_D": overlap / 32,
                "changed_frames_vs_D": 32 - overlap,
                "same_input_as_D": selected == original_indices,
            },
        }
    return plans


def _trial_orders(trials):
    if len(trials) != 6:
        raise ValueError("Require six paired option orders per condition")
    orders = [trial["order"] for trial in trials]
    width = len(orders[0])
    if width not in {3, 4} or any(
        len(order) != width
        or any(type(index) is not int for index in order)
        or sorted(order) != list(range(width))
        for order in orders
    ) or len({tuple(order) for order in orders}) != 6:
        raise ValueError("Require six unique valid option permutations")
    if any(type(trial["scoring_correct"]) is not bool for trial in trials):
        raise ValueError("Correctness must be Boolean")
    return orders


def _resources(trials, arm):
    timings, peaks = [], {name: [] for name in (
        "peak_allocated_bytes", "peak_reserved_bytes"
    )}
    for trial in trials:
        seconds = trial.get("qa_wall_seconds")
        if seconds is not None:
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not np.isfinite(seconds)
                or seconds < 0
            ):
                raise ValueError("Invalid recorded QA wall time")
            timings.append(seconds)
        for key, values in peaks.items():
            value = (trial.get("cuda_memory") or {}).get(key)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid recorded CUDA memory peak")
                values.append(value)
    return {
        "n_trials": len(trials),
        "n_trials_with_timing": len(timings),
        "recorded_qa_seconds_sum": float(sum(timings)) if timings else None,
        "median_qa_seconds": float(np.median(timings)) if timings else None,
        **{
            "max_cuda_" + key: max(values) if values else None
            for key, values in peaks.items()
        },
        "historical_baseline": arm in BASELINES,
        "timing_origin": (
            "historical_source_baseline" if arm in BASELINES
            else "new_condition_recorded_trials_may_be_reused"
        ),
    }


def summarize_rows(rows, seed=42):
    """Aggregate paired six-order means; resample whole videos 5000 times."""
    if not rows:
        raise ValueError("No completed long-ablation questions")
    means, seen = [], set()
    for row in rows:
        identity = (row["video_id"], row["question_id"])
        if (
            any(not isinstance(value, str) or not value for value in identity)
            or identity in seen
            or row.get("duration_group") != "long"
            or row.get("split", "dev") != "dev"
            or set(row["conditions"]) != set(ARMS)
        ):
            raise ValueError("Invalid, duplicate or incomplete dev Long row")
        seen.add(identity)
        reference, correct = None, {}
        for arm in ARMS:
            trials = row["conditions"][arm]["trials"]
            orders = _trial_orders(trials)
            if reference is not None and orders != reference:
                raise ValueError("Conditions have different option orders")
            if "options" in row and len(row["options"]) != len(orders[0]):
                raise ValueError("Option count disagrees with trial orders")
            reference = orders
            correct[arm] = sum(trial["scoring_correct"] for trial in trials)
        values = {arm: correct[arm] / 6 for arm in ARMS}
        deltas = {
            name: correct[first] - correct[second]
            for name, (first, second) in COMPARISONS.items()
        }
        deltas[INTERACTION] = (
            correct["B_topk"] - correct["B"]
            - correct["D_topk"] + correct["D"]
        )
        means.append({
            "video_id": identity[0], "question_id": identity[1],
            "duration_group": "long", "n_option_orders": 6,
            **values,
            **{
                arm + "_correct_orders": value
                for arm, value in correct.items()
            },
            **{key: value / 6 for key, value in deltas.items()},
            **{key + "_numerator": value for key, value in deltas.items()},
        })
    keys = [*ARMS, *COMPARISONS, INTERACTION]
    changes = {
        key: {
            "improved": sum(row[key + "_numerator"] > 0 for row in means),
            "degraded": sum(row[key + "_numerator"] < 0 for row in means),
            "unchanged": sum(row[key + "_numerator"] == 0 for row in means),
        }
        for key in COMPARISONS
    }
    return {
        "experiment": "frozen_D2_B32_long_2x2",
        "frame_budget": 32,
        "split": "dev",
        "duration": "long",
        "conditions": list(ARMS),
        "new_conditions": list(CONDITIONS),
        "primary_comparison": "D_topk_minus_D",
        "metric_definitions": {
            **{key: f"{a} minus {b}" for key, (a, b) in COMPARISONS.items()},
            INTERACTION: "(B_topk - B) - (D_topk - D)",
        },
        "overall": clustered_means(means, keys, seed=seed, samples=5000),
        "paired_question_changes": changes,
        "question_metrics": means,
        "qa_resources": {
            "by_condition": {
                arm: _resources([
                    trial for row in rows
                    for trial in row["conditions"][arm]["trials"]
                ], arm)
                for arm in ARMS
            },
            "scope": (
                "Historical source baselines are separate from new-condition "
                "records. Resume reuses saved trials; each new condition is "
                "evaluated separately even when selected indices match. "
                "Recorded time sums are not fresh inference counts, paired "
                "speed comparisons or elapsed experiment time. QA timings "
                "exclude decoding, CLIP, selection and model loading. CUDA "
                "peaks include resident models in the PyTorch allocator."
            ),
        },
        "interpretation": (
            "Exploratory development after prior holdout inspection. "
            "Within-question six-order means have equal question weight; "
            "5000 paired video-cluster percentile bootstrap draws. The "
            "interaction is a within-question difference of differences, "
            "not subtraction of CI bounds. CIs are marginal without "
            "multiplicity correction; one video cannot estimate uncertainty. "
            "Only allocation policy and within-event selection vary in the "
            "2x2; D2 boundaries remain frozen. CLIP similarity, temporal "
            "coverage and overlap are proxies, not measured evidence recall."
        ),
    }
