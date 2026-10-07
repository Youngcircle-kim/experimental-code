"""Frozen-plan frame-budget selectors and paired, QA-only analysis.

This module does not read files, decode video, load models, or run QA. Event
boundaries and pooled CLIP relevance come from the verified 16-frame plans;
the target budget changes allocation and sampling, not feature extraction.
"""

from copy import deepcopy

import numpy as np

from .ablation import select_frame_control
from .algorithms import allocate_frames, sample_indices
from .bottleneck_diagnostics import validate_plan
from .event_factorial_dev import validate_indices
from .event_refinement import select_inside
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest

ARMS = (
    "uniform",
    "frame_top",
    "temporal_bin",
    "D",
    "D_topk",
    "B_topk",
    "C_topk",
    "C",
)
COMPARISONS = {
    "D_minus_uniform": ("D", "uniform"),
    "D_minus_C": ("D", "C"),
    "D_minus_frame_top": ("D", "frame_top"),
    "D_minus_temporal_bin": ("D", "temporal_bin"),
    "D_topk_minus_D": ("D_topk", "D"),
    "D_topk_minus_C_topk": ("D_topk", "C_topk"),
    "D_topk_minus_B_topk": ("D_topk", "B_topk"),
    "D_topk_minus_uniform": ("D_topk", "uniform"),
    "D_topk_minus_frame_top": ("D_topk", "frame_top"),
    "D_topk_minus_temporal_bin": ("D_topk", "temporal_bin"),
}
BUDGET_COMPARISONS = tuple(f"{a}_32_minus_16" for a in ARMS) + (
    "D_uniform_advantage_change",
)


def make_conditions(d_plan, c_plan, scores, budget=32):
    """Replay frozen C/D partitions, recalculating allocation for the budget.

    The input plans must be valid 16-frame question-relevance plans. No
    answer, option text, caption, or human evidence annotation is used.
    """
    for plan, detector in ((d_plan, "D2"), (c_plan, "D0")):
        validate_plan(plan, 16, "question_relevance")
        if (
            plan["detector"] != detector
            or plan["allocation_policy"] != "question_relevance"
        ):
            raise ValueError("Require saved D2/D0 question-allocation plans")
    count = d_plan["events"][-1][1]
    if c_plan["events"][-1][1] != count or len(c_plan["events"]) != len(
        d_plan["events"]
    ):
        raise ValueError("C and D need matching candidate and event counts")
    if type(budget) is not int or not 1 <= budget <= count:
        raise ValueError("Budget must be an integer within candidate count")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ValueError("Require finite scores for every candidate")

    result = {}
    for arm in ARMS[:3]:
        indices = select_frame_control(arm, scores, budget).tolist()
        validate_indices(indices, count, budget)
        result[arm] = {
            "selected_indices": indices,
            "selected_timestamps": None,
        }
    for arm in ARMS[3:]:
        source = c_plan if arm in ("C", "C_topk") else d_plan
        capacities = np.asarray(source["capacities"], dtype=int)
        relevance = np.asarray(source["relevance_raw_cosine"], dtype=float)
        allocation_scores = (
            np.log(capacities) if arm == "B_topk" else relevance
        )
        temperature = 1.0 if arm == "B_topk" else source["temperature"]
        allocation = allocate_frames(
            allocation_scores, capacities, budget, temperature
        ).tolist()
        if arm.endswith("topk"):
            # With no time gap, candidate indices suffice as dummy times.
            indices, fallback = select_inside(
                scores, np.arange(count), source["events"], allocation, 0.0
            )
            if fallback:
                raise ValueError("Unexpected fallback in unspaced top-k")
        else:
            indices = [
                int(i)
                for (start, stop), n in zip(source["events"], allocation)
                for i in sample_indices(start, stop, n)
            ]
        validate_indices(indices, count, budget)
        weights = np.exp(
            (allocation_scores - allocation_scores.max()) / temperature
        )
        weights /= weights.sum()
        result[arm] = {
            "selected_indices": indices,
            "selected_timestamps": None,
            "detector": source["detector"],
            "events": deepcopy(source["events"]),
            "capacities": capacities.tolist(),
            "relevance_raw_cosine": relevance.tolist(),
            "allocation_scores": allocation_scores.tolist(),
            "temperature": temperature,
            "weights_before_capacity_rounding": weights.tolist(),
            "allocation": allocation,
            "allocation_policy": (
                "candidate_count" if arm == "B_topk" else "question_relevance"
            ),
            "within_event": (
                "question_cosine_topk" if arm.endswith("topk") else "uniform"
            ),
            "min_gap_seconds": 0.0,
            "diagnostics": {
                "zero_allocation_events": sum(n == 0 for n in allocation),
                "multi_frame_events": sum(n > 1 for n in allocation),
                "max_event_allocation": max(allocation),
                "selected_mean_question_cosine": float(scores[indices].mean()),
            },
        }
    d_indices = set(result["D"]["selected_indices"])
    for arm in ARMS[3:]:
        result[arm]["diagnostics"]["overlap_with_D"] = (
            len(d_indices & set(result[arm]["selected_indices"])) / budget
        )
    return result


def _counts(entry, orders, budget):
    trials = entry["trials"]
    if [t["order"] for t in trials] != orders or any(
        type(t["scoring_correct"]) is not bool for t in trials
    ):
        raise ValueError("Require matching six orders and Boolean correctness")
    indices = entry["selected_indices"]
    if not indices:
        raise ValueError("No selected frames")
    validate_indices(indices, indices[-1] + 1, budget)
    return sum(t["scoring_correct"] for t in trials)


def _clustered_rates(count_rows, keys):
    """Bootstrap integer correct-count contrasts before converting to rates.

    All questions have six paired orders. Keeping numerators integral avoids
    artificial tiny positive CI endpoints for an exactly zero count contrast.
    """
    summary = clustered_means(count_rows, keys, seed=42, samples=5000)
    for metric in summary["metrics"].values():
        metric["mean"] /= 6
        if metric["ci95"] is not None:
            metric["ci95"] = [v / 6 for v in metric["ci95"]]
    return summary


def summarize(rows, baseline_rows, protocol):
    """Summarize completed 32-frame QA against matched saved 16-frame QA.

    ``baseline_rows`` maps (video_id, question_id) to a row with all eight
    16-frame arms. Only 32-frame trials originating in this experiment count
    toward new QA cost. Same-input sharing remains counted exactly once.
    """
    if not rows:
        raise ValueError("No completed questions")
    count_rows, baseline_counts, means, unique = [], [], [], {}
    seen = set()
    keys = [*ARMS, *COMPARISONS, *BUDGET_COMPARISONS]
    for row in rows:
        identity = (row["video_id"], row["question_id"])
        if identity in seen or identity not in baseline_rows:
            raise ValueError("Duplicate question or missing 16-frame baseline")
        seen.add(identity)
        baseline = baseline_rows[identity]
        if (
            (baseline["video_id"], baseline["question_id"]) != identity
            or baseline["duration_group"] != row["duration_group"]
            or set(row["conditions"]) != set(ARMS)
            or set(baseline["conditions"]) != set(ARMS)
        ):
            raise ValueError("Incomplete or mismatched paired conditions")
        orders = [t["order"] for t in row["conditions"]["D"]["trials"]]
        if len(orders) != 6 or len({tuple(o) for o in orders}) != 6:
            raise ValueError("Require six distinct paired option orders")
        current, previous, pixel_hashes = {}, {}, {}
        for arm in ARMS:
            entry = row["conditions"][arm]
            current[arm] = _counts(entry, orders, 32)
            previous[arm] = _counts(baseline["conditions"][arm], orders, 16)
            if entry["qa_origin"] != "this_experiment":
                raise ValueError("32-frame QA cannot reuse 16-frame source QA")
            indices = tuple(entry["selected_indices"])
            pixels = entry["qa_pixels_sha256"]
            if indices in pixel_hashes and pixel_hashes[indices] != pixels:
                raise ValueError(
                    "Identical frame indices have different pixels"
                )
            pixel_hashes[indices] = pixels
            for trial in entry["trials"]:
                seconds = trial["qa_wall_seconds"]
                if (
                    isinstance(seconds, bool)
                    or not isinstance(seconds, (int, float))
                    or not np.isfinite(seconds)
                    or seconds < 0
                ):
                    raise ValueError("Invalid QA timing")
                trial_key = (*identity, indices, pixels, tuple(trial["order"]))
                if trial_key in unique and unique[trial_key] != trial:
                    raise ValueError("Identical inputs must share the same QA")
                unique[trial_key] = trial
        values = {
            **current,
            **{
                k: current[a] - current[b] for k, (a, b) in COMPARISONS.items()
            },
            **{f"{a}_32_minus_16": current[a] - previous[a] for a in ARMS},
            "D_uniform_advantage_change": (
                current["D"]
                - current["uniform"]
                - previous["D"]
                + previous["uniform"]
            ),
        }
        labels = {
            k: row[k] for k in ("video_id", "question_id", "duration_group")
        }
        count_rows.append({**labels, **values})
        baseline_counts.append({**labels, **previous})
        means.append({**labels, **{k: v / 6 for k, v in values.items()}})

    resources = {}
    for arm in ARMS:
        entries = [r["conditions"][arm] for r in rows]
        trials = [t for e in entries for t in e["trials"]]
        memory = [
            t["cuda_memory"]["peak_allocated_bytes"]
            for t in trials
            if "cuda_memory" in t
        ]
        if any(
            isinstance(m, bool)
            or not isinstance(m, (int, float))
            or not np.isfinite(m)
            or m < 0
            for m in memory
        ):
            raise ValueError("Invalid QA peak memory")
        resources[arm] = {
            "qa_seconds_median": float(
                np.median([t["qa_wall_seconds"] for t in trials])
            ),
            "max_peak_allocated_GiB": max(memory) / 1024**3
            if memory
            else None,
            "visual_tokens_per_QA_call": 3200,
            "option_order_repeats_per_question": 6,
            "qa_origins": sorted({e["qa_origin"] for e in entries}),
            "selection_and_feature_seconds": None,
        }
    comparison_keys = [*COMPARISONS, *BUDGET_COMPARISONS]
    summary = {
        "stage": "complete",
        "protocol_sha256": report_digest(protocol),
        "evaluation_role": protocol["evaluation_role"],
        "qa_resolution": [320, 320],
        "frame_budget": 32,
        "baseline_frame_budget": 16,
        "compared_frame_budgets": [16, 32],
        "max_segments": 128,
        "primary_comparison": "D_minus_uniform",
        "mechanism_comparisons": [
            "D_minus_C",
            "D_topk_minus_D",
            "D_topk_minus_C_topk",
            "D_topk_minus_B_topk",
        ],
        "budget_comparisons": list(BUDGET_COMPARISONS),
        "overall": _clustered_rates(count_rows, keys),
        "baseline_overall": _clustered_rates(baseline_counts, list(ARMS)),
        "paired_question_changes": {
            k: {
                "improved": sum(r[k] > 0 for r in count_rows),
                "degraded": sum(r[k] < 0 for r in count_rows),
                "unchanged": sum(r[k] == 0 for r in count_rows),
            }
            for k in comparison_keys
        },
        "same_input_as_D_questions": {
            a: sum(
                r["conditions"][a]["selected_indices"]
                == r["conditions"]["D"]["selected_indices"]
                for r in rows
            )
            for a in ARMS
            if a != "D"
        },
        "resources": resources,
        "unique_new_QA_calls": len(unique),
        "unique_new_QA_seconds": sum(
            t["qa_wall_seconds"] for t in unique.values()
        ),
        "cost_scope": protocol["cost_scope"],
        "interpretation": protocol["interpretation"],
    }
    return summary, means
