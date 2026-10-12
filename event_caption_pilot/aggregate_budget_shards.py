"""Validate and pool complete video shards before paired budget analysis."""

from __future__ import annotations

import argparse
import csv
import io
import os
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from .compare_budgets import (
    ARMS,
    FRAME_BUDGETS,
    compare_loaded,
    load_complete,
    normalized_backend,
    normalized_runtime,
)
from .config import Config
from .portable_budget import (
    checkpoint,
    output_lock,
    read_json,
    save_json,
    summarize_rows,
    validate_completed_row,
)
from .qa_diagnostics import report_digest
from .types import Question


def rooted(root, *parts):
    root = Path(root).resolve()
    path = root.joinpath(*parts).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Artifact path must remain inside the suite root")
    return path


def job_directory(root, split, budget, index):
    if (
        split not in {"dev", "eval"}
        or type(budget) is not int
        or budget not in FRAME_BUDGETS
        or type(index) is not int
        or index < 0
    ):
        raise ValueError("Invalid split/budget/shard job identity")
    return rooted(root, "jobs", f"{split}_b{budget}_s{index}")


def expected_protocol(plan, split, budget, index):
    """Derive the exact portable child protocol without forging its digest."""
    entry = plan["splits"][split]
    shards = entry["shards"]
    if (
        type(budget) is not int
        or budget not in plan["budgets"]
        or type(index) is not int
        or not 0 <= index < len(shards)
        or shards[index]["index"] != index
    ):
        raise ValueError("Budget/shard is absent from the suite plan")
    full = entry["protocol"]
    protocol = deepcopy(full)
    protocol["config"]["frame_budget"] = budget
    videos = set(shards[index]["video_ids"])
    protocol["cohort"] = {
        vid: deepcopy(questions)
        for vid, questions in full["cohort"].items()
        if vid in videos
    }
    protocol["inputs"] = {
        vid: deepcopy(identity)
        for vid, identity in full["inputs"].items()
        if vid in videos
    }
    if len(shards) > 1:
        protocol["sharding"] = {
            "index": index,
            "count": len(shards),
            "strategy": "sorted_video_id_round_robin",
            "cohort": deepcopy(full["cohort"]),
        }
    return protocol


def validate_plan(plan):
    if (
        plan.get("version") != 1
        or plan.get("experiment") != "a4500_full_suite_v1"
        or not plan.get("splits")
        or not set(plan["splits"]) <= {"dev", "eval"}
    ):
        raise ValueError("Require a version 1 full-suite plan")
    budgets, gpus = plan["budgets"], plan["gpu_ids"]
    if (
        not budgets
        or any(type(b) is not int or b not in FRAME_BUDGETS for b in budgets)
        or budgets != sorted(set(budgets))
        or len(gpus) != 2
        or len(set(gpus)) != 2
        or any(type(g) is not int or g < 0 for g in gpus)
        or any(
            not Path(plan[k]).is_absolute()
            for k in ("config_path", "manifest_path")
        )
    ):
        raise ValueError("Invalid suite budgets, GPUs or source paths")
    seen_videos, input_files, shared_protocol = set(), {}, None
    for split, entry in plan["splits"].items():
        full, shards = entry["protocol"], entry["shards"]
        cohort = full["cohort"]
        if (
            full.get("experiment") != "portable_frame_budget_v1"
            or full.get("version") != 1
            or full.get("split") != split
            or full.get("max_videos") is not None
            or full.get("max_questions") is not None
            or "sharding" in full
            or not cohort
            or full["config"]["frame_budget"] != max(budgets)
            or full.get("manifest_path") != plan["manifest_path"]
            or set(full["inputs"]) != set(cohort)
            or set(cohort) & seen_videos
            or not 1 <= len(shards) <= len(gpus)
        ):
            raise ValueError(
                "Invalid full protocol or duplicated split cohort"
            )
        shared = {
            key: value
            for key, value in full.items()
            if key not in {"split", "cohort", "inputs"}
        }
        if shared_protocol is not None and shared != shared_protocol:
            raise ValueError(
                "Full split protocols differ outside their cohorts"
            )
        shared_protocol = shared
        seen_videos.update(cohort)
        for vid, questions in cohort.items():
            if (
                not isinstance(vid, str)
                or not vid
                or not questions
                or any(not isinstance(q, str) or not q for q in questions)
                or len(set(questions)) != len(questions)
            ):
                raise ValueError("Invalid full-cohort video/question identity")
            content = full["inputs"][vid].get("sha256")
            if not content or content in input_files:
                raise ValueError(
                    "Missing or duplicate original source file identity"
                )
            input_files[content] = (split, vid)
        covered, assigned_gpus = set(), set()
        for index, shard in enumerate(shards):
            expected = sorted(cohort)[index :: len(shards)]
            if (
                type(shard["index"]) is not int
                or shard["index"] != index
                or type(shard["gpu"]) is not int
                or shard["gpu"] not in gpus
                or shard["gpu"] in assigned_gpus
                or not expected
                or shard["video_ids"] != expected
                or covered & set(shard["video_ids"])
            ):
                raise ValueError(
                    "Invalid, duplicate or nonexhaustive shard assignment"
                )
            covered.update(shard["video_ids"])
            assigned_gpus.add(shard["gpu"])
        if covered != set(cohort):
            raise ValueError("Shards do not cover the full cohort")


def save_csv(path, fieldnames, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text.getvalue())
    os.replace(temporary, path)


def validate_children(root, plan):
    expected_dirs = {
        job_directory(root, split, budget, shard["index"])
        for split, entry in plan["splits"].items()
        for budget in plan["budgets"]
        for shard in entry["shards"]
    }
    jobs = rooted(root, "jobs")
    actual_dirs = {p.resolve() for p in jobs.iterdir() if p.is_dir()}
    if actual_dirs != expected_dirs:
        raise ValueError("Missing or unexpected shard job directories")
    loaded, reference, sources, candidates = {}, None, {}, {}
    for split, entry in plan["splits"].items():
        for budget in plan["budgets"]:
            children, pooled = [], {}
            for shard in entry["shards"]:
                index = shard["index"]
                directory = job_directory(root, split, budget, index)
                for name in (
                    "protocol",
                    "results",
                    "progress",
                    "backend",
                    "runtime",
                ):
                    rooted(root, directory, name + ".json")
                child = load_complete(directory, budget)
                if child["protocol"] != expected_protocol(
                    plan, split, budget, index
                ):
                    raise ValueError(
                        "Child protocol differs from its exact suite plan"
                    )
                if child["runtime"].get("environment", {}).get(
                    "CUDA_VISIBLE_DEVICES"
                ) != str(shard["gpu"]):
                    raise ValueError(
                        "Child ran on the wrong physical GPU mask"
                    )
                backend = child["backend"]
                if backend.get("device") != "cuda:0" or any(
                    backend.get(name, {}).get("devices") != ["cuda:0"]
                    for name in ("encoder_parameters", "vlm_parameters")
                ):
                    raise ValueError(
                        "Child models must reside on its single masked GPU"
                    )
                protocol_hash = report_digest(child["protocol"])
                stored_backend = checkpoint(
                    rooted(root, directory, "backend_identity.json"),
                    {"protocol_sha256": protocol_hash},
                )
                if stored_backend != backend:
                    raise ValueError(
                        "Child backend metadata integrity mismatch"
                    )
                identity = (
                    normalized_backend(backend),
                    normalized_runtime(child["runtime"]),
                )
                if reference is not None and identity != reference:
                    raise ValueError(
                        "Backend/runtime differs across suite children"
                    )
                reference = identity
                if set(pooled) & set(child["rows"]):
                    raise ValueError("Duplicated questions across shards")
                for folder in ("plans", "trials"):
                    target = rooted(root, directory, folder)
                    for path in target.glob("*.json"):
                        rooted(root, path)
                config = Config(**child["protocol"]["config"])
                for row in child["rows"].values():
                    question_id = {
                        "protocol_sha256": protocol_hash,
                        "video_id": row["video_id"],
                        "question_id": row["question_id"],
                    }
                    stored = checkpoint(
                        rooted(
                            root,
                            directory,
                            "questions",
                            report_digest(
                                [
                                    row["video_id"],
                                    row["question_id"],
                                ]
                            )
                            + ".json",
                        ),
                        question_id,
                    )
                    if stored != row:
                        raise ValueError(
                            "Child results/question checkpoint mismatch"
                        )
                    validate_completed_row(
                        row,
                        {
                            "video_id": row["video_id"],
                            "source_id": row["source_id"],
                            "duration": row["duration_group"],
                        },
                        Question(
                            row["question_id"],
                            row["question"],
                            tuple(row["options"]),
                            row["answer_index_evaluator_only"],
                        ),
                        config,
                        protocol_hash,
                        directory,
                    )
                    video = (split, row["video_id"])
                    for lookup, key in (
                        (sources, row["source_id"]),
                        (candidates, row["candidate_sha256"]),
                    ):
                        if key in lookup and lookup[key] != video:
                            raise ValueError(
                                "Source/candidate identity reused across "
                                "videos"
                            )
                        lookup[key] = video
                pooled.update(child["rows"])
                children.append(child)
            expected = {
                (vid, qid)
                for vid, qids in entry["protocol"]["cohort"].items()
                for qid in qids
            }
            if set(pooled) != expected:
                raise ValueError(
                    "Pooled questions do not cover the complete cohort"
                )
            loaded[(split, budget)] = (children, pooled)
    return loaded


def merge_children(root, plan, split, budget, children, rows):
    protocol = deepcopy(plan["splits"][split]["protocol"])
    protocol["experiment"] = "sharded_frame_budget_aggregate_v1"
    protocol["config"]["frame_budget"] = budget
    protocol["aggregation"] = {
        "suite_plan_sha256": report_digest(plan),
        "shard_count": len(children),
        "method": "pool_raw_questions_then_video_cluster_bootstrap",
    }
    ordered = [
        rows[(vid, qid)]
        for vid, qids in protocol["cohort"].items()
        for qid in qids
    ]
    summary = summarize_rows(ordered, protocol["config"]["seed"])
    provenance = {
        "suite_plan_sha256": report_digest(plan),
        "children": [
            {
                "directory": str(child["directory"]),
                "split": split,
                "frame_budget": budget,
                "shard_index": index,
                "gpu": plan["splits"][split]["shards"][index]["gpu"],
                "protocol_sha256": report_digest(child["protocol"]),
                "results_sha256": report_digest(child["results"]),
                "backend_sha256": report_digest(child["backend"]),
                "runtime_sha256": report_digest(child["runtime"]),
            }
            for index, child in enumerate(children)
        ],
    }
    results = {
        "protocol_sha256": report_digest(protocol),
        "completed": True,
        "config": protocol["config"],
        "questions": ordered,
        "summary": summary,
        "provenance": provenance,
    }
    return {
        "directory": rooted(root, "reports", split, f"b{budget}"),
        "protocol": protocol,
        "results": results,
        "rows": rows,
        "backend": children[0]["backend"],
        "runtime": children[0]["runtime"],
    }


def aggregate(root, plan):
    """Validate all cohorts before publishing complete aggregate reports."""
    root = Path(root).resolve()
    with output_lock(rooted(root, ".aggregation")):
        comparison_path = rooted(root, "comparison.json")
        save_json(
            comparison_path, {"status": "validating", "completed": False}
        )
        try:
            validate_plan(plan)
            if read_json(rooted(root, "suite_plan.json")) != plan:
                raise ValueError(
                    "Saved suite plan differs from requested aggregation"
                )
            loaded = validate_children(root, plan)
            merged = {
                key: merge_children(root, plan, *key, *values)
                for key, values in loaded.items()
            }
            pairs = {
                (split, smaller, larger): compare_loaded(
                    merged[(split, smaller)], merged[(split, larger)]
                )
                for split in plan["splits"]
                for smaller, larger in combinations(plan["budgets"], 2)
            }
            compact = {
                "status": "complete",
                "completed": True,
                "experiment": "a4500_full_suite_comparison_v1",
                "version": 1,
                "suite_plan_sha256": report_digest(plan),
                "splits": {},
            }
            accuracy_rows, question_rows = [], []
            for split in plan["splits"]:
                group = {"budgets": {}, "comparisons": {}}
                compact["splits"][split] = group
                for budget in plan["budgets"]:
                    combined = merged[(split, budget)]
                    results, directory = (
                        combined["results"],
                        combined["directory"],
                    )
                    summary = results["summary"]
                    brief = {
                        k: v
                        for k, v in summary.items()
                        if k != "question_metrics"
                    }
                    group["budgets"][str(budget)] = brief
                    save_json(
                        rooted(root, directory, "protocol.json"),
                        combined["protocol"],
                    )
                    save_json(rooted(root, directory, "results.json"), results)
                    save_json(
                        rooted(root, directory, "provenance.json"),
                        results["provenance"],
                    )
                    save_json(
                        rooted(root, directory, "handoff.json"),
                        {
                            "status": "complete",
                            "protocol_sha256": results["protocol_sha256"],
                            "split": split,
                            "frame_budget": budget,
                            "n_questions": summary["overall"]["n_questions"],
                            "n_videos": summary["overall"]["n_videos"],
                            "summary": brief,
                            "provenance": results["provenance"],
                        },
                    )
                    groups = {
                        "overall": summary["overall"],
                        **summary["by_duration"],
                    }
                    for duration, statistics in groups.items():
                        for arm in ARMS:
                            metric = statistics["metrics"][arm]
                            bounds = metric["ci95"] or [None, None]
                            accuracy_rows.append(
                                {
                                    "split": split,
                                    "duration": duration,
                                    "frame_budget": budget,
                                    "arm": arm,
                                    "accuracy": metric["mean"],
                                    "ci95_low": bounds[0],
                                    "ci95_high": bounds[1],
                                    "n_questions": statistics["n_questions"],
                                    "n_videos": statistics["n_videos"],
                                }
                            )
                    for row in summary["question_metrics"]:
                        for arm in ARMS:
                            question_rows.append(
                                {
                                    "split": split,
                                    "frame_budget": budget,
                                    "video_id": row["video_id"],
                                    "question_id": row["question_id"],
                                    "duration_group": row["duration_group"],
                                    "arm": arm,
                                    "accuracy": row[arm],
                                }
                            )
                for smaller, larger in combinations(plan["budgets"], 2):
                    name = f"b{smaller}_b{larger}"
                    comparison = pairs[(split, smaller, larger)]
                    group["comparisons"][name] = {
                        k: v
                        for k, v in comparison.items()
                        if k != "question_metrics"
                    }
                    save_json(
                        rooted(
                            root, "reports", split, f"comparison_{name}.json"
                        ),
                        comparison,
                    )
            save_csv(
                rooted(root, "comparison.csv"),
                list(accuracy_rows[0]),
                accuracy_rows,
            )
            save_csv(
                rooted(root, "question_metrics.csv"),
                list(question_rows[0]),
                question_rows,
            )
            save_json(comparison_path, compact)
            return compact
        except Exception as error:
            save_json(
                comparison_path,
                {
                    "status": "failed",
                    "completed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    plan = read_json(root / "suite_plan.json")
    with output_lock(root):
        aggregate(root, plan)


if __name__ == "__main__":
    main()
