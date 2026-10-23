"""Frozen D2 relevance-by-complexity-by-within-event ablation and audit."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import numpy as np

from .algorithms import allocate_frames, normalize_rows, sample_indices
from .long_ablation_analysis import (
    BASELINES,
    _resources,
    _trial_orders,
)
from .long_ablation_analysis import make_conditions as _original_conditions
from .paired_reanalysis import clustered_means
from .types import Event

CONDITIONS = ("N", "N_topk", "D_topk", "K", "K_topk", "KD", "KD_topk")
ARMS = (*BASELINES, *CONDITIONS)
FRAME_BUDGET = 32
ALGORITHM_SPEC = {
    "version": "frozen-d2-feature-complexity-v1",
    "features": "unit-normalized frozen encoder features",
    "temporal_change": "mean((1-cos(x_i,x_(i+1)))/2) within event",
    "diversity": "n/(n-1)*(1-||mean(x)||^2)/2 for n>1",
    "singleton": "temporal_change=diversity=0",
    "roundoff": (
        "clip half-cosine distances and diversity to [0,1]; zero values "
        "at or below the frozen normalization_epsilon"
    ),
    "raw_complexity": "0.5*temporal_change+0.5*diversity",
    "multiplier": "1+raw_complexity; fixed lambda=1",
    "allocation_log_weights": {
        "N": "0", "D": "relevance/temperature",
        "K": "log(multiplier)",
        "KD": "relevance/temperature+log(multiplier)",
    },
    "numerical_allocation": (
        "center log(multiplier) by its maximum; KD uses relevance + "
        "temperature*centered_log_multiplier at original temperature"
    ),
    "within_event": "uniform or question_cosine_topk; earlier index ties",
    "capacity": "exact 32 unique frames; no duration weight or minimum quota",
    "interpretation": (
        "Bounded feature-variation proxy; not motion, action difficulty or "
        "evidence recall. Relevance off refers to allocation only: topk "
        "still uses the question."
    ),
}
COMPARISONS = {
    "KD_topk_minus_D_topk": ("KD_topk", "D_topk"),
    "KD_minus_D": ("KD", "D"),
    "K_minus_N": ("K", "N"),
    "K_topk_minus_N_topk": ("K_topk", "N_topk"),
    "D_minus_N": ("D", "N"),
    "D_topk_minus_N_topk": ("D_topk", "N_topk"),
    "KD_minus_K": ("KD", "K"),
    "KD_topk_minus_K_topk": ("KD_topk", "K_topk"),
    **{f"{name}_topk_minus_{name}": (f"{name}_topk", name)
       for name in ("N", "D", "K", "KD")},
    **{f"{arm}_minus_{baseline}": (arm, baseline)
       for arm in CONDITIONS
       for baseline in ("uniform", "temporal_bin", "frame_top", "C")},
}
INTERACTIONS = {
    "interaction_uniform": {"KD": 1, "D": -1, "K": -1, "N": 1},
    "interaction_topk": {
        "KD_topk": 1, "D_topk": -1, "K_topk": -1, "N_topk": 1,
    },
    "three_way_interaction": {
        "KD_topk": 1, "D_topk": -1, "K_topk": -1, "N_topk": 1,
        "KD": -1, "D": 1, "K": 1, "N": -1,
    },
}


def _integers(values, name):
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain one-dimensional integers")
    return result


def _vector(values, count, name):
    result = np.asarray(values, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite {name} vector")
    return result


def _close(actual, expected, name):
    actual, expected = (np.asarray(x, dtype=float) for x in (actual, expected))
    if (
        actual.shape != expected.shape or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or not np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError(f"Saved {name} does not reproduce")


def _bounds(events, count):
    if isinstance(events, (list, tuple)) and events and isinstance(
        events[0], Event,
    ):
        events = [[event.start, event.stop] for event in events]
    bounds = np.asarray(events)
    if (
        bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
        or bounds.dtype.kind not in {"i", "u"}
        or bounds[0, 0] != 0 or bounds[-1, 1] != count
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(bounds[1:, 0] != bounds[:-1, 1])
    ):
        raise ValueError("Events must partition all candidates exactly")
    return bounds.astype(np.int64, copy=False)


def _feature_digest(features):
    values = np.ascontiguousarray(features, dtype=np.float64)
    return sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


def event_complexity(features, events, epsilon=1e-8):
    """Compute query-independent variation once per video's frozen events.

    Diversity equals the mean half-cosine distance over distinct frame pairs,
    computed in O(n*d), without an n-by-n similarity matrix. Boundary jumps
    between events never contribute to temporal variation.
    """
    source_hash = _feature_digest(features)
    visual = normalize_rows(features, epsilon)
    bounds = _bounds(events, len(visual))
    temporal, diversity, sums, mean_norms = [], [], [], []
    capacities = bounds[:, 1] - bounds[:, 0]
    for (start, stop), number in zip(bounds, capacities):
        part = visual[start:stop]
        average = part.mean(axis=0)
        norm_squared = float(np.clip(average @ average, 0, 1))
        distance_sum = 0.0
        temporal_value = diversity_value = 0.0
        if number > 1:
            distances = np.clip(
                (1 - np.sum(part[:-1] * part[1:], axis=1)) / 2, 0, 1,
            )
            distances[distances <= epsilon] = 0
            distance_sum = float(distances.sum())
            temporal_value = distance_sum / (number - 1)
            diversity_value = float(np.clip(
                number / (number - 1) * (1 - norm_squared) / 2, 0, 1,
            ))
            if diversity_value <= epsilon:
                diversity_value = 0.0
        sums.append(distance_sum)
        mean_norms.append(norm_squared)
        temporal.append(float(temporal_value))
        diversity.append(diversity_value)
    raw = (np.asarray(temporal) + np.asarray(diversity)) / 2
    return {
        "spec": deepcopy(ALGORITHM_SPEC),
        "normalization_epsilon": float(epsilon),
        "source_features_sha256": source_hash,
        "events": bounds.tolist(), "capacities": capacities.tolist(),
        "adjacent_distance_sum": sums,
        "mean_feature_norm_squared": mean_norms,
        "temporal_change": temporal, "diversity": diversity,
        "raw": raw.tolist(), "multiplier": (1 + raw).tolist(),
    }


def _validate_complexity(saved, bounds, epsilon):
    if saved.get("spec") != ALGORITHM_SPEC:
        raise ValueError("Unknown complexity algorithm specification")
    if saved.get("normalization_epsilon") != epsilon:
        raise ValueError("Complexity normalization tolerance changed")
    if saved.get("events") != bounds.tolist():
        raise ValueError("Complexity events disagree with frozen D")
    capacities = bounds[:, 1] - bounds[:, 0]
    if not np.array_equal(
        _integers(saved["capacities"], "complexity capacities"), capacities,
    ):
        raise ValueError("Complexity capacities disagree with frozen D")
    digest = saved.get("source_features_sha256")
    if (
        not isinstance(digest, str) or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("Invalid complexity source feature identity")
    size = len(bounds)
    distance_sum = _vector(saved["adjacent_distance_sum"], size, "change sum")
    norm_squared = _vector(
        saved["mean_feature_norm_squared"], size, "mean squared norm",
    )
    if (
        np.any(distance_sum < 0) or np.any(distance_sum > capacities - 1)
        or np.any(norm_squared < 0) or np.any(norm_squared > 1)
    ):
        raise ValueError("Complexity sufficient statistics exceed bounds")
    temporal, diversity = np.zeros(size), np.zeros(size)
    multiple = capacities > 1
    temporal[multiple] = distance_sum[multiple] / (capacities[multiple] - 1)
    diversity[multiple] = np.clip(
        capacities[multiple] / (capacities[multiple] - 1)
        * (1 - norm_squared[multiple]) / 2, 0, 1,
    )
    diversity[diversity <= epsilon] = 0
    _close(saved["temporal_change"], temporal, "temporal complexity")
    _close(saved["diversity"], diversity, "feature diversity")
    raw = (temporal + diversity) / 2
    _close(saved["raw"], raw, "raw complexity")
    _close(saved["multiplier"], 1 + raw, "complexity multiplier")
    return raw


def _allocation(name, relevance, raw, capacities, temperature):
    base = name.removesuffix("_topk")
    log_multiplier = np.log1p(raw)
    centered = log_multiplier - log_multiplier.max()
    if base == "N":
        scores, temp, log_weights = np.zeros(len(raw)), 1.0, np.zeros(len(raw))
    elif base == "D":
        scores, temp = relevance, temperature
        log_weights = relevance / temperature
    elif base == "K":
        scores, temp, log_weights = centered, 1.0, log_multiplier
    elif base == "KD":
        scores, temp = relevance + temperature * centered, temperature
        log_weights = relevance / temperature + log_multiplier
    else:
        raise ValueError("Unknown complexity condition")
    allocation = allocate_frames(scores, capacities, FRAME_BUDGET, temp)
    weights = np.exp((scores - scores.max()) / temp)
    weights /= weights.sum()
    return scores, float(temp), log_weights, weights, allocation


def _choose(bounds, allocation, frame_scores, topk):
    selected = []
    for (start, stop), number in zip(bounds, allocation):
        if topk:
            candidates = np.arange(start, stop)
            order = np.lexsort((candidates, -frame_scores[candidates]))
            chosen = candidates[order[:int(number)]]
        else:
            chosen = sample_indices(int(start), int(stop), int(number))
        selected.extend(int(i) for i in chosen)
    return sorted(selected)


def _diagnostics(selected, original, allocation, capacities, weights,
                 times, features, frame_scores):
    bins = np.empty(len(times), dtype=int)
    for index, part in enumerate(np.array_split(np.arange(len(times)), 32)):
        bins[part] = index
    positive = weights[weights > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    overlap = len(set(selected) & set(original))
    visual = features[selected]
    return {
        "n_events": len(allocation),
        "events_exceed_budget": len(allocation) > 32,
        "zero_allocation_events": int((allocation == 0).sum()),
        "zero_allocation_fraction": float(np.mean(allocation == 0)),
        "event_coverage_fraction": float(np.mean(allocation > 0)),
        "unallocated_candidate_fraction": float(
            capacities[allocation == 0].sum() / len(times),
        ),
        "multi_frame_events": int((allocation > 1).sum()),
        "max_event_allocation": int(allocation.max()),
        "weight_entropy": entropy,
        "normalized_weight_entropy": (
            entropy / float(np.log(len(allocation)))
            if len(allocation) > 1 else 0.0
        ),
        "effective_weighted_events": float(np.exp(entropy)),
        "fixed_bin_coverage": len(set(bins[selected])) / 32,
        "max_candidate_time_gap_seconds": float(
            np.diff(np.r_[times[0], times[selected], times[-1]]).max(),
        ),
        "selected_mean_question_cosine": float(frame_scores[selected].mean()),
        "selected_feature_redundancy": float(
            ((visual @ visual.T).sum() - 32) / (32 * 31),
        ),
        "overlap_with_D": overlap / 32,
        "changed_frames_vs_D": 32 - overlap,
        "same_input_as_D": selected == original,
    }


def make_conditions(features, query, times, d_plan, config, *,
                    precomputed_complexity=None):
    """Freeze events and D quotas, then cross relevance, complexity and topk.

    Passing event_complexity's result avoids recomputing visual variation for
    each question. The cache is tied to the exact feature matrix and events.
    """
    # This also verifies saved D events, relevance, integer quotas, uniform
    # selection and timestamps against the original normalization sequence.
    anchor = _original_conditions(features, query, times, d_plan, config)[
        "D_topk"
    ]
    complexity = (
        event_complexity(
            features, d_plan["events"], config.normalization_epsilon,
        )
        if precomputed_complexity is None else precomputed_complexity
    )
    visual = normalize_rows(features, config.normalization_epsilon)
    query = normalize_rows(
        np.asarray(query)[None, :], config.normalization_epsilon,
    )[0]
    times = np.asarray(times, dtype=float)
    bounds = _bounds(d_plan["events"], len(visual))
    raw = _validate_complexity(
        complexity, bounds, config.normalization_epsilon,
    )
    if complexity["source_features_sha256"] != _feature_digest(features):
        raise ValueError("Complexity belongs to different source features")
    capacities = bounds[:, 1] - bounds[:, 0]
    # The old helper verifies numerical agreement with the replayed encoder.
    # Allocation uses the exact historical values, preserving frozen R even
    # when an accepted tiny replay drift would perturb an integer quota tie.
    relevance = np.asarray(d_plan["relevance_raw_cosine"], dtype=float)
    frame_scores = visual @ query
    plans = {}
    for name in CONDITIONS:
        scores, temp, log_weights, weights, allocation = _allocation(
            name, relevance, raw, capacities, config.allocation_temperature,
        )
        if name == "D_topk" and allocation.tolist() != d_plan["allocation"]:
            raise ValueError("D_topk changed original D allocation")
        topk = name.endswith("_topk")
        selected = _choose(bounds, allocation, frame_scores, topk)
        plans[name] = {
            "detector": "D2", "condition": name,
            "allocation_policy": {
                "N": "equal_event", "D": "question_relevance",
                "K": "feature_complexity",
                "KD": "question_relevance_times_feature_complexity",
            }[name.removesuffix("_topk")],
            "within_event": "question_cosine_topk" if topk else "uniform",
            "events": bounds.tolist(), "capacities": capacities.tolist(),
            "complexity": deepcopy(complexity),
            "relevance_raw_cosine": relevance.tolist(),
            "frame_scores_raw_cosine": frame_scores.tolist(),
            "allocation_log_weights": log_weights.tolist(),
            "allocation_scores": scores.tolist(), "temperature": temp,
            "weights_before_capacity_rounding": weights.tolist(),
            "allocation": allocation.tolist(),
            "selected_indices": selected,
            "selected_timestamps": times[selected].tolist(),
            "diagnostics": _diagnostics(
                selected, d_plan["selected_indices"], allocation, capacities,
                weights, times, visual, frame_scores,
            ),
        }
        diagnostics = plans[name]["diagnostics"]
        diagnostics["event_candidate_spans_seconds"] = [
            float(times[stop - 1] - times[start]) for start, stop in bounds
        ]
        diagnostics["singleton_events"] = int((capacities == 1).sum())
        diagnostics["zero_complexity_events"] = int((raw == 0).sum())
        diagnostics["allocation_changed_vs_D"] = (
            allocation.tolist() != d_plan["allocation"]
        )
        comparison_ids = (
            anchor["selected_indices"] if topk else d_plan["selected_indices"]
        )
        diagnostics["same_input_as_relevance_only_matched_within"] = (
            selected == comparison_ids
        )
        diagnostics["changed_frames_vs_relevance_only_matched_within"] = (
            32 - len(set(selected) & set(comparison_ids))
        )
        validate_selection(plans[name], d_plan, name, times, config)
    return plans


def validate_selection(selection, frozen_d, name, times, config):
    """Audit saved components -> weights -> exact quotas and frame indices.

    Does not claim to re-encode the source video: feature/pixel identity and
    model provenance are checked by the worker. Persisted per-frame scores
    make topk choices fully auditable without model inference.
    """
    config.validate()
    if name not in CONDITIONS or config.frame_budget != FRAME_BUDGET:
        raise ValueError("Unknown condition or non-B32 complexity protocol")
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1 or len(times) < 32 or not np.isfinite(times).all()
        or times[0] < 0 or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Invalid candidate timestamps")
    bounds = _bounds(frozen_d["events"], len(times))
    capacities = bounds[:, 1] - bounds[:, 0]
    if (
        frozen_d.get("detector") != "D2"
        or frozen_d.get("allocation_policy") != "question_relevance"
        or frozen_d.get("temperature") != config.allocation_temperature
        or frozen_d.get("capacities") != capacities.tolist()
        or selection.get("events") != bounds.tolist()
        or selection.get("capacities") != capacities.tolist()
        or selection.get("condition") != name
        or selection.get("detector") != "D2"
    ):
        raise ValueError("Selection does not match frozen D events/protocol")
    base = name.removesuffix("_topk")
    expected_policy = {
        "N": "equal_event", "D": "question_relevance",
        "K": "feature_complexity",
        "KD": "question_relevance_times_feature_complexity",
    }[base]
    topk = name.endswith("_topk")
    if (
        selection.get("allocation_policy") != expected_policy
        or selection.get("within_event") != (
            "question_cosine_topk" if topk else "uniform"
        )
    ):
        raise ValueError("Selection policy does not match its condition")
    raw = _validate_complexity(
        selection["complexity"], bounds, config.normalization_epsilon,
    )
    relevance = _vector(
        frozen_d["relevance_raw_cosine"], len(bounds), "frozen relevance",
    )
    _close(frozen_d["allocation_scores"], relevance, "frozen D scores")
    _close(selection["relevance_raw_cosine"], relevance, "event relevance")
    original = allocate_frames(
        relevance, capacities, FRAME_BUDGET, config.allocation_temperature,
    )
    if not np.array_equal(
        _integers(frozen_d["allocation"], "frozen allocation"), original,
    ):
        raise ValueError("Frozen D allocation does not reproduce")
    original_ids = _choose(bounds, original, np.zeros(len(times)), False)
    if (
        frozen_d["selected_indices"] != original_ids
        or frozen_d["selected_timestamps"] != times[original_ids].tolist()
    ):
        raise ValueError("Frozen D uniform selection does not reproduce")
    scores, temp, log_weights, weights, allocation = _allocation(
        name, relevance, raw, capacities, config.allocation_temperature,
    )
    _close(selection["allocation_scores"], scores, "allocation scores")
    _close(selection["allocation_log_weights"], log_weights, "log weights")
    _close(selection["weights_before_capacity_rounding"], weights, "weights")
    if (
        type(selection.get("temperature")) not in {int, float}
        or selection["temperature"] != temp
        or not np.array_equal(
            _integers(selection["allocation"], "allocation"), allocation,
        )
    ):
        raise ValueError("Saved complexity allocation does not reproduce")
    frame_scores = _vector(
        selection["frame_scores_raw_cosine"], len(times), "frame cosine",
    )
    if np.any(np.abs(frame_scores) > 1 + 1e-12):
        raise ValueError("Frame cosine scores exceed [-1,1]")
    selected = _integers(selection["selected_indices"], "selected indices")
    expected = _choose(bounds, allocation, frame_scores, topk)
    if (
        len(selected) != 32 or len(set(selected.tolist())) != 32
        or selected.tolist() != expected
        or selection["selected_timestamps"] != times[expected].tolist()
    ):
        raise ValueError("Saved within-event selection does not reproduce")


def _treatment_exposure(rows):
    """Separate absent treatment after rounding from negative effects."""
    comparisons = {}
    for first, second in (
        ("KD", "D"), ("KD_topk", "D_topk"),
        ("K", "N"), ("K_topk", "N_topk"),
    ):
        observed, quota_changed, indices_changed, replaced = 0, 0, 0, []
        changed_videos = set()
        for row in rows:
            left, right = (row["conditions"][arm] for arm in (first, second))
            if any(
                field not in condition for condition in (left, right)
                for field in ("allocation", "selected_indices")
            ):
                continue
            observed += 1
            quota_changed += left["allocation"] != right["allocation"]
            changed = left["selected_indices"] != right["selected_indices"]
            indices_changed += changed
            if changed:
                changed_videos.add(row["video_id"])
            replaced.append(32 - len(
                set(left["selected_indices"]) & set(right["selected_indices"]),
            ))
        comparisons[f"{first}_versus_{second}"] = {
            "questions_with_selection_records": observed,
            "questions_missing_selection_records": len(rows) - observed,
            "allocation_changed_questions": quota_changed,
            "allocation_changed_fraction": (
                quota_changed / observed if observed else None
            ),
            "selected_input_changed_questions": indices_changed,
            "selected_input_changed_fraction": (
                indices_changed / observed if observed else None
            ),
            "videos_with_changed_selected_input": len(changed_videos),
            "mean_replaced_frames": float(np.mean(replaced)) if replaced
            else None,
        }
    zeros = {}
    by_video = {}
    for arm in ARMS:
        allocations = [row["conditions"][arm]["allocation"] for row in rows
                       if "allocation" in row["conditions"][arm]]
        zeros[arm] = {
            "questions_with_allocation_records": len(allocations),
            "zero_allocated_event_question_pairs": sum(
                sum(value == 0 for value in allocation)
                for allocation in allocations
            ),
            "event_question_pairs": sum(map(len, allocations)),
        }
    for row in rows:
        condition = row["conditions"]["KD"]
        if "complexity" not in condition:
            continue
        record = {
            "complexity": condition["complexity"],
            "candidate_spans_seconds": condition.get("diagnostics", {}).get(
                "event_candidate_spans_seconds",
            ),
        }
        vid = row["video_id"]
        if vid in by_video and by_video[vid] != record:
            raise ValueError("Complexity changed between questions of a video")
        by_video[vid] = record
    return {
        "comparisons": comparisons, "zero_allocation_by_condition": zeros,
        "event_complexity_by_video": by_video,
        "scope": (
            "Selection changes are computed before QA. A bounded multiplier "
            "may leave integer allocation and inputs unchanged. Missing "
            "selection records are reported, never treated as no change."
        ),
    }


def summarize_rows(rows, seed=42):
    """Use paired six-order question means and 5000 whole-video resamples."""
    if not rows:
        raise ValueError("No completed complexity-ablation questions")
    means, seen = [], set()
    for row in rows:
        identity = (row["video_id"], row["question_id"])
        if (
            any(not isinstance(value, str) or not value for value in identity)
            or identity in seen or row.get("duration_group") != "long"
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
        deltas = {key: correct[a] - correct[b]
                  for key, (a, b) in COMPARISONS.items()}
        deltas.update({
            key: sum(weight * correct[arm] for arm, weight in terms.items())
            for key, terms in INTERACTIONS.items()
        })
        means.append({
            "video_id": identity[0], "question_id": identity[1],
            "duration_group": "long", "n_option_orders": 6,
            **{arm: value / 6 for arm, value in correct.items()},
            **{arm + "_correct_orders": value
               for arm, value in correct.items()},
            **{key: value / 6 for key, value in deltas.items()},
            **{key + "_numerator": value for key, value in deltas.items()},
        })
    keys = [*ARMS, *COMPARISONS, *INTERACTIONS]
    return {
        "experiment": "frozen_D2_B32_long_relevance_complexity_within_2x2x2",
        "algorithm_spec": deepcopy(ALGORITHM_SPEC),
        "frame_budget": 32, "split": "dev", "duration": "long",
        "conditions": list(ARMS), "new_conditions": list(CONDITIONS),
        "primary_comparison": "KD_topk_minus_D_topk",
        "metric_definitions": {
            **{key: f"{a} minus {b}" for key, (a, b) in COMPARISONS.items()},
            "interaction_uniform": "(KD - D) - (K - N)",
            "interaction_topk": "(KD_topk - D_topk) - (K_topk - N_topk)",
            "three_way_interaction": "interaction_topk - interaction_uniform",
        },
        "overall": clustered_means(means, keys, seed=seed, samples=5000),
        "treatment_exposure": _treatment_exposure(rows),
        "paired_question_changes": {
            key: {
                "improved": sum(row[key + "_numerator"] > 0 for row in means),
                "degraded": sum(row[key + "_numerator"] < 0 for row in means),
                "unchanged": sum(
                    row[key + "_numerator"] == 0 for row in means
                ),
            } for key in COMPARISONS
        },
        "question_metrics": means,
        "qa_resources": {
            "by_condition": {
                arm: _resources([
                    trial for row in rows
                    for trial in row["conditions"][arm]["trials"]
                ], arm) for arm in ARMS
            },
            "scope": (
                "Historical source baselines are separate from new-condition "
                "records. Resume reuses saved trials; conditions are "
                "evaluated "
                "separately even when selected indices match. Recorded sums "
                "are not fresh inference counts, paired speed comparisons or "
                "elapsed experiment time. QA time excludes video decoding, "
                "encoder, complexity, selection and model loading. CUDA peaks "
                "include resident models in the PyTorch allocator."
            ),
        },
        "interpretation": (
            "Exploratory dev analysis after prior holdout inspection. D2 "
            "boundaries and candidate pool stay frozen; eight factorial arms "
            "cross allocation relevance, feature complexity and within-event "
            "topk. N is equal-event weighting, not candidate-count weighting. "
            "Question-conditioned topk remains active when allocation "
            "relevance "
            "is off. Six-order means receive equal question weight; 5000 "
            "paired video-cluster percentile bootstrap draws. Interactions "
            "are computed within questions before resampling. CIs are "
            "marginal "
            "without multiplicity correction; one video has no estimable CI. "
            "Feature variation, similarity and coverage are proxies; they do "
            "not measure action difficulty or ground-truth evidence recall."
        ),
    }
