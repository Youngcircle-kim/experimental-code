"""Count-matched D2 versus fixed partitions at B32, with paired audits."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from itertools import combinations

import numpy as np

from .algorithms import event_features, normalize_rows, segment
from .complexity_ablation_analysis import (
    ALGORITHM_SPEC as COMPLEXITY_SPEC,
)
from .complexity_ablation_analysis import (
    _allocation,
    _bounds,
    _choose,
    _close,
    _diagnostics,
    _feature_digest,
    _integers,
    _validate_complexity,
    _vector,
    event_complexity,
)
from .long_ablation_analysis import BASELINES, _resources, _trial_orders
from .paired_reanalysis import clustered_means

CAPS = (16, 24, 32)
SEGMENTATIONS = ("event", "fixed")
VARIANTS = ("D", "D_topk", "KD", "KD_topk")
CONDITIONS = tuple(
    f"{kind}_k{cap}_{variant}"
    for cap in CAPS
    for kind in SEGMENTATIONS
    for variant in VARIANTS
)
ARMS = (*BASELINES, *CONDITIONS)
FRAME_BUDGET = 32
ALGORITHM_SPEC = {
    "version": "count-matched-d2-fixed-b32-v1",
    "caps": list(CAPS),
    "event_partition": (
        "D2 with max_segments replaced by cap; threshold, minimum length "
        "and window unchanged; no forced split when actual count is below cap"
    ),
    "fixed_partition": (
        "D0 equal candidate-index bins matching the actual D2 count for "
        "each video and cap, not the nominal cap"
    ),
    "relevance": "cosine of normalized mean event feature and question",
    "normalization": (
        "Preserve portable_budget.make_selection normalization: one pass "
        "for frame ranking, a second pass before event_features "
        "and event query"
    ),
    "complexity": deepcopy(COMPLEXITY_SPEC),
    "allocation_variants": ["D", "KD"],
    "within_event": ["uniform", "question_cosine_topk"],
    "budget": "exact 32 unique candidates, no minimum per-event quota",
    "inference": (
        "Exploratory dev Long only; paired six-order question means; 5000 "
        "video-cluster resamples; marginal 95% intervals without multiplicity "
        "correction; cap comparisons are not guaranteed actual-count changes"
    ),
}


def condition_spec(name):
    """Decode a fixed condition name without guessing from arbitrary text."""
    if name not in CONDITIONS:
        raise ValueError(f"Unknown segment-count condition: {name}")
    kind, cap, variant = name.split("_", 2)
    return {
        "partition": f"{kind}_{cap}",
        "segmentation": kind,
        "event_cap": int(cap[1:]),
        "allocation": variant.removesuffix("_topk"),
        "within_event": (
            "question_cosine_topk" if variant.endswith("_topk") else "uniform"
        ),
    }


COMPARISONS = {
    **{
        f"event_minus_fixed_k{cap}_{variant}": (
            f"event_k{cap}_{variant}",
            f"fixed_k{cap}_{variant}",
        )
        for cap in CAPS
        for variant in VARIANTS
    },
    **{
        f"{kind}_k{cap}_KD_minus_D{suffix}": (
            f"{kind}_k{cap}_KD{suffix}",
            f"{kind}_k{cap}_D{suffix}",
        )
        for cap in CAPS
        for kind in SEGMENTATIONS
        for suffix in ("", "_topk")
    },
    **{
        f"{kind}_k{cap}_{base}_topk_minus_uniform": (
            f"{kind}_k{cap}_{base}_topk",
            f"{kind}_k{cap}_{base}",
        )
        for cap in CAPS
        for kind in SEGMENTATIONS
        for base in ("D", "KD")
    },
    **{
        f"{kind}_cap{small}_minus_cap{large}_{variant}": (
            f"{kind}_k{small}_{variant}",
            f"{kind}_k{large}_{variant}",
        )
        for small, large in combinations(CAPS, 2)
        for kind in SEGMENTATIONS
        for variant in VARIANTS
    },
}
SELECTION_COMPARISONS = dict(COMPARISONS)
HISTORICAL_COMPARISONS = {
    **{
        f"event_k{cap}_D_minus_historical_D": (f"event_k{cap}_D", "D")
        for cap in CAPS
    },
    **{
        f"event_k{cap}_KD_topk_minus_{baseline}": (
            f"event_k{cap}_KD_topk",
            baseline,
        )
        for cap in CAPS
        for baseline in ("uniform", "frame_top", "temporal_bin")
    },
}
COMPARISONS.update(HISTORICAL_COMPARISONS)


def _settings(config, cap):
    return {
        "boundary_threshold": config.boundary_threshold,
        "min_segment_frames": config.min_segment_frames,
        "window_size": config.window_size,
        "normalization_epsilon": config.normalization_epsilon,
        "max_segments": cap,
    }


def _config(config):
    config.validate()
    if config.frame_budget != FRAME_BUDGET:
        raise ValueError("Segment-count ablation requires exactly B32")


def _times(times):
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < FRAME_BUDGET
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Require at least 32 chronological candidate timestamps"
        )
    return times


def prepare_partitions(features, config):
    """Compute six question-independent partitions and complexity records."""
    _config(config)
    visual = normalize_rows(features, config.normalization_epsilon)
    if len(visual) < FRAME_BUDGET:
        raise ValueError("Require at least 32 unique candidates")
    event_input = normalize_rows(visual, config.normalization_epsilon)
    digest = _feature_digest(features)
    prepared = {}
    for cap in CAPS:
        capped = replace(config, max_segments=cap)
        detected = segment(features, capped, "D2")
        partitions = {
            "event": detected,
            "fixed": segment(
                features, capped, "D0", segment_count=len(detected)
            ),
        }
        for kind, events in partitions.items():
            name = f"{kind}_k{cap}"
            bounds = _bounds(events, len(visual))
            prepared[name] = {
                "partition": name,
                "segmentation": kind,
                "event_cap": cap,
                "actual_event_count": len(events),
                "detector": "D2" if kind == "event" else "D0",
                "source_features_sha256": digest,
                "segmentation_settings": _settings(config, cap),
                "events": bounds.tolist(),
                "capacities": (bounds[:, 1] - bounds[:, 0]).tolist(),
                "complexity": event_complexity(
                    features,
                    events,
                    config.normalization_epsilon,
                ),
                "pooled_features": event_features(
                    event_input,
                    events,
                    config.normalization_epsilon,
                ).tolist(),
            }
    return prepared


def _validate_partition(record, name, count, config):
    spec = condition_spec(name)
    bounds = _bounds(record["events"], count)
    actual = len(bounds)
    cap = spec["event_cap"]
    if (
        record.get("partition") != spec["partition"]
        or record.get("segmentation") != spec["segmentation"]
        or record.get("event_cap") != cap
        or type(record.get("event_cap")) is not int
        or record.get("actual_event_count") != actual
        or type(record.get("actual_event_count")) is not int
        or actual > cap
        or record.get("detector")
        != ("D2" if spec["segmentation"] == "event" else "D0")
        or record.get("segmentation_settings") != _settings(config, cap)
        or record.get("capacities") != (bounds[:, 1] - bounds[:, 0]).tolist()
        or (
            count >= config.min_segment_frames
            and np.any(bounds[:, 1] - bounds[:, 0] < config.min_segment_frames)
        )
    ):
        raise ValueError("Partition metadata, cap or minimum length changed")
    if spec["segmentation"] == "fixed":
        edges = [index * count // actual for index in range(actual + 1)]
        if bounds.tolist() != [
            list(pair) for pair in zip(edges[:-1], edges[1:])
        ]:
            raise ValueError("Fixed bins are not equal candidate-index bins")
    raw = _validate_complexity(
        record["complexity"],
        bounds,
        config.normalization_epsilon,
    )
    if (
        record.get("source_features_sha256")
        != record["complexity"]["source_features_sha256"]
    ):
        raise ValueError(
            "Partition and complexity feature identities disagree"
        )
    return bounds, raw


def make_conditions(
    features, query_features, times, config, *, partitions=None
):
    """Construct all 24 conditions; options and gold never enter selection."""
    _config(config)
    times = _times(times)
    visual = normalize_rows(features, config.normalization_epsilon)
    query = normalize_rows(
        np.asarray(query_features)[None, :],
        config.normalization_epsilon,
    )[0]
    event_query = normalize_rows(query[None, :], config.normalization_epsilon)[
        0
    ]
    if len(visual) != len(times) or query.shape != (visual.shape[1],):
        raise ValueError("Candidate, question and timestamp shapes disagree")
    if partitions is None:
        partitions = prepare_partitions(features, config)
    if set(partitions) != {
        f"{kind}_k{cap}" for cap in CAPS for kind in SEGMENTATIONS
    }:
        raise ValueError("Incomplete cached partitions")
    digest = _feature_digest(features)
    frame_scores = visual @ query
    plans = {}
    for cap in CAPS:
        if (
            partitions[f"event_k{cap}"]["actual_event_count"]
            != partitions[f"fixed_k{cap}"]["actual_event_count"]
        ):
            raise ValueError("Fixed bins must match actual event count")
    for name in CONDITIONS:
        spec = condition_spec(name)
        record = partitions[spec["partition"]]
        bounds, raw = _validate_partition(record, name, len(times), config)
        if record["source_features_sha256"] != digest:
            raise ValueError(
                "Cached partitions belong to different source features"
            )
        pooled = np.asarray(record["pooled_features"], dtype=float)
        if (
            pooled.shape != (len(bounds), len(query))
            or not np.isfinite(pooled).all()
            or not np.allclose(np.linalg.norm(pooled, axis=1), 1, atol=1e-12)
        ):
            raise ValueError("Invalid cached pooled event features")
        relevance = pooled @ event_query
        capacities = bounds[:, 1] - bounds[:, 0]
        variant = name.split("_", 2)[2]
        scores, temp, logs, weights, allocation = _allocation(
            variant,
            relevance,
            raw,
            capacities,
            config.allocation_temperature,
        )
        topk = spec["within_event"] == "question_cosine_topk"
        selected = _choose(bounds, allocation, frame_scores, topk)
        original = _allocation(
            "D",
            relevance,
            raw,
            capacities,
            config.allocation_temperature,
        )[-1]
        original_ids = _choose(bounds, original, frame_scores, False)
        plans[name] = {
            key: deepcopy(value)
            for key, value in record.items()
            if key != "pooled_features"
        }
        plans[name].update(
            {
                "condition": name,
                "allocation_policy": (
                    "question_relevance"
                    if spec["allocation"] == "D"
                    else "question_relevance_times_feature_complexity"
                ),
                "within_event": spec["within_event"],
                "relevance_raw_cosine": relevance.tolist(),
                "frame_scores_raw_cosine": frame_scores.tolist(),
                "allocation_log_weights": logs.tolist(),
                "allocation_scores": scores.tolist(),
                "temperature": temp,
                "weights_before_capacity_rounding": weights.tolist(),
                "allocation": allocation.tolist(),
                "selected_indices": selected,
                "selected_timestamps": times[selected].tolist(),
                "diagnostics": _diagnostics(
                    selected,
                    original_ids,
                    allocation,
                    capacities,
                    weights,
                    times,
                    visual,
                    frame_scores,
                ),
            }
        )
        diagnostics = plans[name]["diagnostics"]
        diagnostics.update(
            {
                "event_cap": spec["event_cap"],
                "actual_event_count": len(bounds),
                "actual_count_below_cap": len(bounds) < spec["event_cap"],
                "one_frame_events": int((allocation == 1).sum()),
                "multi_frame_event_fraction": float(np.mean(allocation >= 2)),
                "frames_in_multi_frame_events": int(
                    allocation[allocation >= 2].sum()
                ),
                "event_candidate_spans_seconds": [
                    float(times[stop - 1] - times[start])
                    for start, stop in bounds
                ],
                "zero_complexity_events": int((raw == 0).sum()),
                "allocation_changed_vs_D": allocation.tolist()
                != original.tolist(),
                "changed_frames_vs_relevance_only_matched_within": 32
                - len(
                    set(selected)
                    & set(_choose(bounds, original, frame_scores, topk))
                ),
            }
        )
        validate_selection(plans[name], name, times, config)
    return plans


def validate_selection(saved, name, times, config):
    """Replay weights, integer quotas and selected frames from saved scores.

    Boundary detection and encoder fidelity require the worker's frozen-source
    replay. This check does not re-encode video or prove semantic scores.
    """
    _config(config)
    times = _times(times)
    spec = condition_spec(name)
    bounds, raw = _validate_partition(saved, name, len(times), config)
    expected_policy = (
        "question_relevance"
        if spec["allocation"] == "D"
        else "question_relevance_times_feature_complexity"
    )
    if (
        saved.get("condition") != name
        or saved.get("within_event") != spec["within_event"]
        or saved.get("allocation_policy") != expected_policy
    ):
        raise ValueError("Selection condition or policy changed")
    relevance = _vector(
        saved["relevance_raw_cosine"], len(bounds), "relevance"
    )
    frame_scores = _vector(
        saved["frame_scores_raw_cosine"], len(times), "frame scores"
    )
    if np.any(np.abs(relevance) > 1 + 1e-12) or np.any(
        np.abs(frame_scores) > 1 + 1e-12
    ):
        raise ValueError("Cosine scores exceed [-1,1]")
    scores, temp, logs, weights, allocation = _allocation(
        name.split("_", 2)[2],
        relevance,
        raw,
        bounds[:, 1] - bounds[:, 0],
        config.allocation_temperature,
    )
    _close(saved["allocation_scores"], scores, "allocation scores")
    _close(saved["allocation_log_weights"], logs, "allocation log weights")
    _close(
        saved["weights_before_capacity_rounding"],
        weights,
        "allocation weights",
    )
    if (
        type(saved.get("temperature")) not in {int, float}
        or saved["temperature"] != temp
        or not np.array_equal(
            _integers(saved["allocation"], "allocation"), allocation
        )
    ):
        raise ValueError("Saved integer allocation does not reproduce")
    selected = _integers(saved["selected_indices"], "selected indices")
    expected = _choose(
        bounds,
        allocation,
        frame_scores,
        spec["within_event"] == "question_cosine_topk",
    )
    if (
        len(selected) != 32
        or len(set(selected.tolist())) != 32
        or selected.tolist() != expected
        or saved["selected_timestamps"] != times[expected].tolist()
    ):
        raise ValueError("Saved selected frames do not reproduce")


def summarize_selections(video_plans):
    """Summarize treatment exposure before QA, without labels."""
    if not video_plans:
        raise ValueError("No video selection plans")
    rows, videos, seen = [], {}, set()
    for plan in video_plans:
        vid = plan["video_id"]
        if (
            not isinstance(vid, str)
            or not vid
            or vid in seen
            or not plan["questions"]
        ):
            raise ValueError("Invalid or repeated video selection plan")
        seen.add(vid)
        reference = {}
        for qid, conditions in plan["questions"].items():
            if (
                not isinstance(qid, str)
                or not qid
                or set(conditions) != set(CONDITIONS)
            ):
                raise ValueError("Incomplete question selection plan")
            for name, selection in conditions.items():
                spec = condition_spec(name)
                key = spec["partition"]
                frozen = {
                    field: selection[field]
                    for field in (
                        "events",
                        "complexity",
                        "actual_event_count",
                        "source_features_sha256",
                    )
                }
                if key in reference and reference[key] != frozen:
                    raise ValueError(
                        "Partition or complexity changed across "
                        "questions/conditions"
                    )
                reference[key] = frozen
                allocation = _integers(selection["allocation"], "allocation")
                selected = _integers(
                    selection["selected_indices"], "selected indices"
                )
                if (
                    len(allocation) != selection["actual_event_count"]
                    or np.any(allocation < 0)
                    or allocation.sum() != 32
                    or len(selected) != 32
                    or len(set(selected.tolist())) != 32
                ):
                    raise ValueError(
                        "Invalid selection budget in diagnostic records"
                    )
            for cap in CAPS:
                if (
                    reference[f"event_k{cap}"]["actual_event_count"]
                    != reference[f"fixed_k{cap}"]["actual_event_count"]
                ):
                    raise ValueError(
                        "Event and fixed actual counts are not matched"
                    )
            rows.append(
                {"video_id": vid, "question_id": qid, "conditions": conditions}
            )
        videos[vid] = {
            f"k{cap}": {
                "cap": cap,
                "actual_event_count": reference[f"event_k{cap}"][
                    "actual_event_count"
                ],
                "actual_count_below_cap": reference[f"event_k{cap}"][
                    "actual_event_count"
                ]
                < cap,
                "same_boundaries_event_fixed": reference[f"event_k{cap}"][
                    "events"
                ]
                == reference[f"fixed_k{cap}"]["events"],
            }
            for cap in CAPS
        }
    occupancy = {}
    for arm in CONDITIONS:
        allocations = [
            np.asarray(row["conditions"][arm]["allocation"]) for row in rows
        ]
        all_counts = np.concatenate(allocations)
        occupancy[arm] = {
            "n_questions": len(rows),
            "event_question_pairs": int(len(all_counts)),
            "zero_frame_event_question_pairs": int((all_counts == 0).sum()),
            "one_frame_event_question_pairs": int((all_counts == 1).sum()),
            "multi_frame_event_question_pairs": int((all_counts >= 2).sum()),
            "zero_frame_fraction": float(np.mean(all_counts == 0)),
            "one_frame_fraction": float(np.mean(all_counts == 1)),
            "multi_frame_fraction": float(np.mean(all_counts >= 2)),
            "questions_with_any_multi_frame_event": sum(
                bool(np.any(a >= 2)) for a in allocations
            ),
            "mean_frames_in_multi_frame_events": float(
                np.mean([a[a >= 2].sum() for a in allocations])
            ),
            "mean_actual_event_count": float(
                np.mean([len(a) for a in allocations])
            ),
        }
    changes = {}
    for key, (a, b) in SELECTION_COMPARISONS.items():
        boundaries, inputs, quotas, replaced = [], [], [], []
        changed_videos = set()
        for row in rows:
            left, right = (row["conditions"][arm] for arm in (a, b))
            same_partition = left["events"] == right["events"]
            boundaries.append(not same_partition)
            inputs.append(
                left["selected_indices"] != right["selected_indices"]
            )
            # Indexwise quota equality requires identical events.
            if same_partition:
                quotas.append(left["allocation"] != right["allocation"])
            replaced.append(
                32
                - len(
                    set(left["selected_indices"])
                    & set(right["selected_indices"])
                )
            )
            if inputs[-1]:
                changed_videos.add(row["video_id"])
        changes[key] = {
            "left": a,
            "right": b,
            "n_questions": len(rows),
            "partition_changed_questions": sum(boundaries),
            "same_partition_questions": len(rows) - sum(boundaries),
            "allocation_changed_among_same_partition_questions": sum(quotas),
            "selected_input_changed_questions": sum(inputs),
            "selected_input_changed_fraction": float(np.mean(inputs)),
            "same_selected_input_questions": len(rows) - sum(inputs),
            "videos_with_changed_selected_input": len(changed_videos),
            "mean_replaced_frames": float(np.mean(replaced)),
        }
    return {
        "n_videos": len(videos),
        "n_questions": len(rows),
        "actual_counts_by_video": videos,
        "counts_by_cap": {
            f"k{cap}": {
                "cap": cap,
                "actual_event_counts": [
                    videos[vid][f"k{cap}"]["actual_event_count"]
                    for vid in sorted(videos)
                ],
                "videos_below_cap": sum(
                    v[f"k{cap}"]["actual_count_below_cap"]
                    for v in videos.values()
                ),
                "actual_count_histogram": {
                    str(count): sum(
                        v[f"k{cap}"]["actual_event_count"] == count
                        for v in videos.values()
                    )
                    for count in sorted(
                        {
                            v[f"k{cap}"]["actual_event_count"]
                            for v in videos.values()
                        }
                    )
                },
            }
            for cap in CAPS
        },
        "allocation_occupancy_by_condition": occupancy,
        "comparisons": changes,
        "adjacent_cap_duplicates": {
            key: value
            for key, value in changes.items()
            if "cap16_minus_cap24" in key or "cap24_minus_cap32" in key
        },
        "scope": (
            "Computed before QA; counts are actual D2 and paired fixed-bin "
            "counts. A cap need not be reached or change boundaries/inputs. "
            "Lower counts do not guarantee multiple frames in every event."
        ),
    }


def summarize_rows(rows, config):
    """Use six paired orders per question and resample whole videos."""
    _config(config)
    if not rows:
        raise ValueError("No completed segment-count questions")
    metrics, seen, videos = [], set(), defaultdict(dict)
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
                raise ValueError(
                    "Conditions have different paired option orders"
                )
            if "options" in row and len(row["options"]) != len(orders[0]):
                raise ValueError("Option count disagrees with trial orders")
            reference = orders
            correct[arm] = sum(trial["scoring_correct"] for trial in trials)
        deltas = {
            key: correct[a] - correct[b] for key, (a, b) in COMPARISONS.items()
        }
        metrics.append(
            {
                "video_id": identity[0],
                "question_id": identity[1],
                "duration_group": "long",
                "n_option_orders": 6,
                **{arm: value / 6 for arm, value in correct.items()},
                **{
                    arm + "_correct_orders": value
                    for arm, value in correct.items()
                },
                **{key: value / 6 for key, value in deltas.items()},
                **{key + "_numerator": value for key, value in deltas.items()},
            }
        )
        videos[identity[0]][identity[1]] = {
            arm: row["conditions"][arm] for arm in CONDITIONS
        }
    return {
        "experiment": "dev_long_B32_count_matched_event_fixed_complexity",
        "algorithm_spec": deepcopy(ALGORITHM_SPEC),
        "frame_budget": 32,
        "split": "dev",
        "duration": "long",
        "conditions": list(ARMS),
        "new_conditions": list(CONDITIONS),
        "historical_reference_conditions": list(BASELINES),
        "comparison_families": {
            "matched_boundaries": [
                key
                for key in COMPARISONS
                if key.startswith("event_minus_fixed")
            ],
            "complexity_effect": [
                key for key in COMPARISONS if "KD_minus_D" in key
            ],
            "within_event_choice": [
                key for key in COMPARISONS if "topk_minus_uniform" in key
            ],
            "exploratory_cap_effect": [
                key for key in COMPARISONS if "_cap" in key
            ],
            "exploratory_historical_reference": list(HISTORICAL_COMPARISONS),
        },
        "metric_definitions": {
            key: f"{a} minus {b}" for key, (a, b) in COMPARISONS.items()
        },
        "overall": clustered_means(
            metrics, [*ARMS, *COMPARISONS], seed=config.seed, samples=5000
        ),
        "selection_diagnostics": summarize_selections(
            [
                {"video_id": vid, "questions": questions}
                for vid, questions in sorted(videos.items())
            ]
        ),
        "paired_question_changes": {
            key: {
                "improved": sum(
                    row[key + "_numerator"] > 0 for row in metrics
                ),
                "degraded": sum(
                    row[key + "_numerator"] < 0 for row in metrics
                ),
                "unchanged": sum(
                    row[key + "_numerator"] == 0 for row in metrics
                ),
            }
            for key in COMPARISONS
        },
        "question_metrics": metrics,
        "qa_resources": {
            "by_condition": {
                arm: _resources(
                    [
                        trial
                        for row in rows
                        for trial in row["conditions"][arm]["trials"]
                    ],
                    arm,
                )
                for arm in ARMS
            },
            "scope": (
                "Recorded QA trials include resumed records; sums are not "
                "elapsed experiment time. QA time excludes decoding, encoder, "
                "partition, "
                "complexity, selection and model loading. CUDA peaks include "
                "resident models in the PyTorch allocator."
            ),
        },
        "interpretation": (
            "Exploratory dev after prior results; no new confirmatory set. "
            "Paired marginal 95% CIs use 5000 video-cluster resamples with "
            "no multiplicity correction. Six orders are repeated measures. "
            "Event versus fixed includes boundary, capacity, pooled relevance "
            "and complexity changes. Nominal cap contrasts may have identical "
            "actual partitions; inspect pre-QA treatment exposure."
        ),
    }
