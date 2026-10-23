"""Two-GPU dev Long B32 event complexity x relevance x selection study."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from .a4500_full_suite import execute_workers
from .a4500_long_ablation import (
    separate_output,
    validate_launch,
    verify_inputs,
)
from .aggregate_budget_shards import rooted, save_csv
from .complexity_ablation_analysis import (
    ALGORITHM_SPEC,
    ARMS,
    BASELINES,
    CONDITIONS,
    summarize_rows,
)
from .long_ablation_source import (
    build_snapshot,
    code_hashes,
    validate_snapshot,
    verify_source_code,
)
from .portable_budget import output_lock, read_json, save_json
from .qa_diagnostics import report_digest

EXPERIMENT = "a4500_complexity_ablation_v1"


def validate_frozen(plan, snapshot):
    """Seal the recipe, full cohort and original two-GPU assignment."""
    validate_snapshot(snapshot)
    expected = {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_root": snapshot["source_root"],
        "snapshot_sha256": report_digest(snapshot),
        "algorithm": ALGORITHM_SPEC,
        "conditions": list(CONDITIONS),
        "qa_side": 320,
        "frame_budget": 32,
        "split": "dev",
        "duration": "long",
        "shards": snapshot["source_shards"],
        "n_videos": len(snapshot["cohort"]),
        "n_questions": len(snapshot["baseline_rows"]),
        "expected_new_qa_trials": len(snapshot["baseline_rows"]) * 42,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Complexity plan/snapshot/algorithm mismatch")
    gpus, shards = plan["gpu_ids"], plan["shards"]
    if (
        len(gpus) != 2
        or len(set(gpus)) != 2
        or len(shards) != 2
        or any(type(g) is not int or g < 0 for g in gpus)
        or not plan.get("code_sha256")
    ):
        raise ValueError("Require two distinct original GPU IDs and code")
    covered = []
    for index, shard in enumerate(shards):
        if (
            shard["index"] != index
            or shard["gpu"] != gpus[index]
            or not shard["video_ids"]
            or shard["video_ids"] != sorted(set(shard["video_ids"]))
        ):
            raise ValueError("Invalid frozen complexity GPU shard")
        covered.extend(shard["video_ids"])
    if len(covered) != len(set(covered)) or set(covered) != set(
        snapshot["cohort"]
    ):
        raise ValueError("Complexity shards must cover every source video")


def build_plan(snapshot, gpu_ids=(0, 1)):
    plan = {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_root": snapshot["source_root"],
        "snapshot_sha256": report_digest(snapshot),
        "algorithm": deepcopy(ALGORITHM_SPEC),
        "gpu_ids": list(gpu_ids),
        "shards": deepcopy(snapshot["source_shards"]),
        "conditions": list(CONDITIONS),
        "qa_side": 320,
        "frame_budget": 32,
        "split": "dev",
        "duration": "long",
        "n_videos": len(snapshot["cohort"]),
        "n_questions": len(snapshot["baseline_rows"]),
        "expected_new_qa_trials": len(snapshot["baseline_rows"]) * 42,
        "code_sha256": code_hashes(),
    }
    validate_frozen(plan, snapshot)
    return plan


def read_frozen(plan_path, expected_digest=None, verify_code=True):
    path = Path(plan_path).resolve()
    plan = read_json(path)
    if expected_digest is not None and report_digest(plan) != expected_digest:
        raise ValueError("Frozen complexity plan digest changed")
    snapshot = read_json(rooted(path.parent, "source_snapshot.json"))
    validate_frozen(plan, snapshot)
    if verify_code:
        if plan["code_sha256"] != code_hashes():
            raise ValueError("Code changed after complexity planning")
        for protocol in snapshot["source_protocols"].values():
            verify_source_code(protocol["code_sha256"])
    return plan, snapshot


def freeze_inputs(root, plan, snapshot):
    root = Path(root).resolve()
    separate_output(root, plan["source_root"])
    validate_frozen(plan, snapshot)
    plan_path = rooted(root, "suite_plan.json")
    source_path = rooted(root, "source_snapshot.json")
    if plan_path.exists():
        if read_frozen(plan_path) != (plan, snapshot):
            raise ValueError("Complexity inputs changed; use a new output")
        return
    if root.exists() and any(
        p.name not in {".run.lock", "source_snapshot.json"}
        for p in root.iterdir()
    ):
        raise ValueError("Output has no compatible complexity plan")
    if source_path.exists() and read_json(source_path) != snapshot:
        raise ValueError("Existing complexity source snapshot differs")
    save_json(source_path, snapshot)
    save_json(plan_path, plan)


def worker_specs(plan, root):
    from .complexity_ablation_worker import job_directory

    root = Path(root).resolve()
    saved, snapshot = read_frozen(rooted(root, "suite_plan.json"))
    if saved != plan:
        raise ValueError("Requested complexity jobs differ from frozen plan")
    config = snapshot["config"]
    return [
        {
            "id": job_directory(root, shard["index"]).name,
            "gpu": shard["gpu"],
            "shard_index": shard["index"],
            "output": str(job_directory(root, shard["index"])),
            "log": str(
                rooted(
                    root,
                    "logs",
                    job_directory(root, shard["index"]).name + ".log",
                )
            ),
            "command": [
                sys.executable,
                "-u",
                "-m",
                "event_caption_pilot.complexity_ablation_worker",
                "--plan",
                str(rooted(root, "suite_plan.json")),
                "--shard-index",
                str(shard["index"]),
                "--expected-plan-sha256",
                report_digest(plan),
            ],
            "environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(shard["gpu"]),
                "PYTHONHASHSEED": str(config["seed"]),
                "OMP_NUM_THREADS": "4",
                "CUBLAS_WORKSPACE_CONFIG": config["cublas_workspace_config"],
            },
        }
        for shard in plan["shards"]
    ]


def aggregate(root, plan=None):
    from .complexity_ablation_worker import job_directory, validate_worker

    root = Path(root).resolve()
    saved, snapshot = read_frozen(
        rooted(root, "suite_plan.json"), verify_code=False
    )
    if plan is not None and plan != saved:
        raise ValueError("Aggregate plan differs from frozen complexity plan")
    plan = saved
    jobs = rooted(root, "jobs")
    expected = {job_directory(root, s["index"]) for s in plan["shards"]}
    if (
        not jobs.exists()
        or {p.resolve() for p in jobs.iterdir() if p.is_dir()} != expected
    ):
        raise ValueError("Missing or unexpected complexity worker folders")
    fresh = {}
    for shard in plan["shards"]:
        for row in validate_worker(root, plan, snapshot, shard["index"]):
            key = row["video_id"], row["question_id"]
            if key in fresh:
                raise ValueError("Duplicate complexity question across shards")
            fresh[key] = row
    baseline = {
        (r["video_id"], r["question_id"]): r for r in snapshot["baseline_rows"]
    }
    if set(fresh) != set(baseline):
        raise ValueError(
            "Complexity results do not cover all source questions"
        )
    rows = []
    for key in sorted(baseline):
        row = deepcopy(baseline[key])
        row["baseline_execution_order"] = row.pop("execution_order", None)
        row["new_execution_order"] = fresh[key]["execution_order"]
        row["conditions"].update(deepcopy(fresh[key]["conditions"]))
        rows.append(row)
    result = {
        "version": 1,
        "experiment": "a4500_complexity_ablation_comparison_v1",
        "status": "complete",
        "completed": True,
        "suite_plan_sha256": report_digest(plan),
        "snapshot_sha256": plan["snapshot_sha256"],
        "algorithm": deepcopy(plan["algorithm"]),
        "split": "dev",
        "duration": "long",
        "frame_budget": 32,
        "qa_side": 320,
        "n_videos": plan["n_videos"],
        "n_questions": plan["n_questions"],
        "new_conditions": list(CONDITIONS),
        "new_qa_trials": sum(
            len(r["conditions"][arm]["trials"])
            for r in rows
            for arm in CONDITIONS
        ),
        "baseline_qa_trials_reused": plan["n_questions"] * len(BASELINES) * 6,
        "summary": summarize_rows(rows, seed=snapshot["config"]["seed"]),
        "questions": rows,
    }
    if result["new_qa_trials"] != plan["expected_new_qa_trials"]:
        raise ValueError("Complexity QA trial count differs from full plan")
    save_json(rooted(root, "comparison.json"), result)
    save_json(
        rooted(root, "handoff.json"),
        {k: v for k, v in result.items() if k != "questions"},
    )
    csv_rows = [
        {
            "split": "dev",
            "frame_budget": 32,
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "duration_group": "long",
            "arm": arm,
            "accuracy": sum(
                t["scoring_correct"] for t in row["conditions"][arm]["trials"]
            )
            / 6,
            "resource_origin": (
                "new_condition_record"
                if arm in CONDITIONS
                else "historical_baseline"
            ),
        }
        for row in rows
        for arm in ARMS
    ]
    save_csv(
        rooted(root, "question_accuracy.csv"), list(csv_rows[0]), csv_rows
    )
    return result


def save_status(root, status):
    for name in ("suite_status.json", "comparison.json", "handoff.json"):
        save_json(rooted(root, name), status)


def run_suite(args, plan, snapshot):
    validate_launch(args)
    root = args.output_root.resolve()
    separate_output(root, args.source_root)
    with output_lock(root):
        freeze_inputs(root, plan, snapshot)
        status = {
            "status": "running",
            "completed": False,
            "suite_plan_sha256": report_digest(plan),
            "completed_jobs": [],
        }
        save_status(root, status)

        def progress(completed):
            status["completed_jobs"] = completed
            save_json(rooted(root, "suite_status.json"), status)

        try:
            verify_inputs(snapshot)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "event_caption_pilot.a4500_suite",
                    "check",
                    "--gpu-ids",
                    *(str(g) for g in args.gpu_ids),
                    "--minimum-free-gib",
                    str(args.minimum_free_gib),
                ],
                env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                check=True,
            )
            status["completed_jobs"] = execute_workers(
                worker_specs(plan, root), on_complete=progress
            )
            aggregate(root, plan)
            status.update(status="complete", completed=True)
            save_json(rooted(root, "suite_status.json"), status)
        except BaseException as error:
            status.update(status="failed", completed=False, error=str(error))
            save_status(root, status)
            raise
    print(f"Complete: {root / 'handoff.json'}", flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("check", "run", "summarize"))
    result.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/a4500_siglip2_long_baseline"),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/a4500_siglip2_complexity"),
    )
    result.add_argument("--gpu-ids", nargs=2, type=int, default=[0, 1])
    result.add_argument("--minimum-free-gib", type=float, default=16.0)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == "summarize":
        identity = read_json(rooted(args.output_root, "suite_plan.json"))
        if identity.get("experiment") != EXPERIMENT:
            raise ValueError("Output is not a complexity experiment")
        with output_lock(args.output_root):
            try:
                report = aggregate(args.output_root)
            except BaseException as error:
                save_status(
                    args.output_root,
                    {
                        "status": "failed",
                        "completed": False,
                        "suite_plan_sha256": report_digest(identity),
                        "error": str(error),
                    },
                )
                raise
            save_json(
                rooted(args.output_root, "suite_status.json"),
                {
                    "status": "complete",
                    "completed": True,
                    "suite_plan_sha256": report["suite_plan_sha256"],
                },
            )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "questions"}, indent=2
            )
        )
        return 0
    validate_launch(args)
    separate_output(args.output_root, args.source_root)
    snapshot = build_snapshot(args.source_root)
    plan = build_plan(snapshot, args.gpu_ids)
    if args.stage == "check":
        verify_inputs(snapshot)
        print(
            json.dumps(
                {
                    "status": "inputs_validated_no_inference",
                    **{
                        k: plan[k]
                        for k in (
                            "n_videos",
                            "n_questions",
                            "frame_budget",
                            "qa_side",
                            "expected_new_qa_trials",
                            "conditions",
                            "shards",
                            "algorithm",
                        )
                    },
                },
                indent=2,
            )
        )
        return 0
    return run_suite(args, plan, snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
