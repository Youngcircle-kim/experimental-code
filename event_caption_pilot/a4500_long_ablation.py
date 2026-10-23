"""Two-GPU, frozen-B32 dev Long allocation x within-event selection study."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from .a4500_full_suite import execute_workers
from .aggregate_budget_shards import rooted, save_csv
from .long_ablation_analysis import ARMS, CONDITIONS, summarize_rows
from .long_ablation_source import (
    build_snapshot,
    code_hashes,
    read_frozen,
    validate_frozen,
)
from .portable_budget import file_identity, output_lock, read_json, save_json
from .qa_diagnostics import report_digest


def build_plan(snapshot, gpu_ids=(0, 1)):
    # Preserve baseline GPU assignments as well as the within-video pairing.
    gpus = list(gpu_ids)
    if gpus != [s["gpu"] for s in snapshot["source_shards"]]:
        raise ValueError("Use the same GPU IDs/order as the source B32 shards")
    plan = {
        "version": 1,
        "experiment": "a4500_long_ablation_v1",
        "source_root": snapshot["source_root"],
        "snapshot_sha256": report_digest(snapshot),
        "gpu_ids": gpus,
        "shards": deepcopy(snapshot["source_shards"]),
        "conditions": list(CONDITIONS),
        "qa_side": 320,
        "split": "dev",
        "duration": "long",
        "frame_budget": 32,
        "n_videos": len(snapshot["cohort"]),
        "n_questions": len(snapshot["baseline_rows"]),
        "expected_new_qa_trials": len(snapshot["baseline_rows"]) * 18,
        "code_sha256": code_hashes(),
    }
    validate_frozen(plan, snapshot)
    return plan


def separate_output(root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("Output root must be separate from the source suite")


def freeze_inputs(root, plan, snapshot):
    root = Path(root).resolve()
    separate_output(root, plan["source_root"])
    validate_frozen(plan, snapshot)
    source_path = rooted(root, "source_snapshot.json")
    plan_path = rooted(root, "suite_plan.json")
    if plan_path.exists():
        saved_plan, saved_source = read_frozen(plan_path)
        if saved_plan != plan or saved_source != snapshot:
            raise ValueError("Inputs/settings changed; use a new output root")
        return
    if root.exists() and any(
        p.name not in {".run.lock", "source_snapshot.json"}
        for p in root.iterdir()
    ):
        raise ValueError("Existing output has no compatible ablation plan")
    if source_path.exists() and read_json(source_path) != snapshot:
        raise ValueError("Existing source snapshot differs from this run")
    save_json(source_path, snapshot)
    save_json(plan_path, plan)


def worker_specs(plan, root):
    root = Path(root).resolve()
    snapshot = read_json(rooted(root, "source_snapshot.json"))
    config = snapshot["config"]
    specs = []
    for shard in plan["shards"]:
        index, gpu = shard["index"], shard["gpu"]
        job_id = f"dev_long_b32_s{index}"
        specs.append(
            {
                "id": job_id,
                "gpu": gpu,
                "shard_index": index,
                "output": str(rooted(root, "jobs", job_id)),
                "log": str(rooted(root, "logs", job_id + ".log")),
                "command": [
                    sys.executable,
                    "-u",
                    "-m",
                    "event_caption_pilot.long_ablation_worker",
                    "--plan",
                    str(rooted(root, "suite_plan.json")),
                    "--shard-index",
                    str(index),
                    "--expected-plan-sha256",
                    report_digest(plan),
                ],
                "environment": {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONHASHSEED": str(config["seed"]),
                    "OMP_NUM_THREADS": "4",
                    "CUBLAS_WORKSPACE_CONFIG": config[
                        "cublas_workspace_config"
                    ],
                },
            }
        )
    return specs


def verify_inputs(snapshot):
    """Check original movies before starting either GPU worker."""
    for index, (vid, identity) in enumerate(snapshot["inputs"].items(), 1):
        print(
            f"Checking original video {index}/{len(snapshot['inputs'])}: "
            f"{vid}",
            flush=True,
        )
        current = file_identity(Path(identity["path"]))
        if any(current[k] != identity[k] for k in ("size_bytes", "sha256")):
            raise ValueError(f"Original video content changed: {vid}")


def validate_launch(args):
    # Preflight and workers must refer to the same physical card indices.
    # An empty mask hides every GPU, so it also needs to be removed.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError(
            "Run `unset CUDA_VISIBLE_DEVICES` before launching the suite; "
            "choose the physical cards with --gpu-ids"
        )
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib <= 0:
        raise ValueError("minimum-free-gib must be finite and positive")
    if len(set(args.gpu_ids)) != 2 or min(args.gpu_ids) < 0:
        raise ValueError("Select two distinct nonnegative GPU IDs")


def aggregate(root, plan=None):
    """Offline aggregation using the sealed baseline and trial checkpoints."""
    from .long_ablation_worker import validate_worker

    root = Path(root).resolve()
    saved, snapshot = read_frozen(
        rooted(root, "suite_plan.json"), verify_code=False
    )
    if plan is not None and plan != saved:
        raise ValueError("Aggregate plan differs from the frozen plan")
    plan = saved
    expected_jobs = {
        rooted(root, "jobs", f"dev_long_b32_s{s['index']}")
        for s in plan["shards"]
    }
    jobs = rooted(root, "jobs")
    if (
        not jobs.exists()
        or {p.resolve() for p in jobs.iterdir() if p.is_dir()} != expected_jobs
    ):
        raise ValueError("Missing or unexpected ablation worker directories")
    fresh = {}
    for shard in plan["shards"]:
        for row in validate_worker(root, plan, snapshot, shard["index"]):
            key = row["video_id"], row["question_id"]
            if key in fresh:
                raise ValueError("Duplicate ablation question across shards")
            fresh[key] = row
    baseline = {
        (r["video_id"], r["question_id"]): r for r in snapshot["baseline_rows"]
    }
    if set(fresh) != set(baseline):
        raise ValueError("New conditions do not cover all baseline questions")
    rows = []
    for key in sorted(baseline):
        row = deepcopy(baseline[key])
        row["baseline_execution_order"] = row.pop("execution_order", None)
        row["new_execution_order"] = fresh[key].get("execution_order")
        row["conditions"].update(deepcopy(fresh[key]["conditions"]))
        rows.append(row)
    stats = summarize_rows(rows, seed=snapshot["config"]["seed"])
    result = {
        "version": 1,
        "experiment": "a4500_long_ablation_comparison_v1",
        "status": "complete",
        "completed": True,
        "suite_plan_sha256": report_digest(plan),
        "snapshot_sha256": plan["snapshot_sha256"],
        "split": "dev",
        "duration": "long",
        "frame_budget": 32,
        "qa_side": 320,
        "n_videos": plan["n_videos"],
        "n_questions": plan["n_questions"],
        "new_conditions": list(CONDITIONS),
        "new_qa_trials": sum(
            len(r["conditions"][n]["trials"]) for r in rows for n in CONDITIONS
        ),
        "baseline_qa_trials_reused": plan["n_questions"] * 5 * 6,
        "summary": stats,
        "questions": rows,
    }
    if result["new_qa_trials"] != plan["expected_new_qa_trials"]:
        raise ValueError("Wrong number of new-condition QA trials")
    save_json(rooted(root, "comparison.json"), result)
    # Keep the handoff compact; full selections, raw orders, times and CUDA
    # peaks remain in comparison.json and the validated worker checkpoints.
    save_json(
        rooted(root, "handoff.json"),
        {k: v for k, v in result.items() if k != "questions"},
    )
    csv_rows = []
    for row in rows:
        for name in ARMS:
            trials = row["conditions"][name]["trials"]
            csv_rows.append(
                {
                    "split": "dev",
                    "frame_budget": 32,
                    "video_id": row["video_id"],
                    "question_id": row["question_id"],
                    "duration_group": "long",
                    "arm": name,
                    "accuracy": sum(t["scoring_correct"] for t in trials) / 6,
                    "resource_origin": "new_run"
                    if name in CONDITIONS
                    else "historical_baseline",
                }
            )
    save_csv(
        rooted(root, "question_accuracy.csv"), list(csv_rows[0]), csv_rows
    )
    return result


def run_suite(args, plan, snapshot):
    validate_launch(args)
    root = args.output_root.resolve()
    separate_output(root, args.source_root)
    rooted(root, ".run.lock")
    with output_lock(root):
        freeze_inputs(root, plan, snapshot)
        status = {
            "status": "running",
            "completed": False,
            "suite_plan_sha256": report_digest(plan),
            "completed_jobs": [],
        }
        save_json(rooted(root, "suite_status.json"), status)
        save_json(rooted(root, "comparison.json"), status)
        save_json(rooted(root, "handoff.json"), status)

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
            for name in (
                "suite_status.json",
                "comparison.json",
                "handoff.json",
            ):
                save_json(rooted(root, name), status)
            raise
    print(f"Complete: {root / 'handoff.json'}", flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("check", "run", "summarize"))
    result.add_argument(
        "--source-root", type=Path, default=Path("outputs/a4500_full_min64")
    )
    result.add_argument(
        "--output-root", type=Path, default=Path("outputs/a4500_long_ablation")
    )
    result.add_argument("--gpu-ids", nargs=2, type=int, default=[0, 1])
    result.add_argument("--minimum-free-gib", type=float, default=16.0)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == "summarize":
        identity = read_json(rooted(args.output_root, "suite_plan.json"))
        if identity.get("experiment") != "a4500_long_ablation_v1":
            raise ValueError("Output root is not a Long ablation experiment")
        rooted(args.output_root, ".run.lock")
        with output_lock(args.output_root):
            try:
                report = aggregate(args.output_root)
            except BaseException as error:
                for name in (
                    "suite_status.json",
                    "comparison.json",
                    "handoff.json",
                ):
                    save_json(
                        rooted(args.output_root, name),
                        {
                            "status": "failed",
                            "completed": False,
                            "error": str(error),
                            "suite_plan_sha256": report_digest(identity),
                        },
                    )
                raise
            save_json(
                rooted(args.output_root, "suite_status.json"),
                {
                    "status": "complete",
                    "completed": True,
                    "suite_plan_sha256": report["suite_plan_sha256"],
                    "completed_jobs": [
                        f"dev_long_b32_s{s['index']}"
                        for s in identity["shards"]
                    ],
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
