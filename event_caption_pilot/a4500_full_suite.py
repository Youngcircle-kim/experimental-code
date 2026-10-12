"""Compare all manifest questions at 16/32/64 frames on two isolated GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from . import portable_budget as runner
from .aggregate_budget_shards import (
    aggregate,
    expected_protocol,
    job_directory,
)
from .config import Config
from .data import array_digest, decode_video, validate_video
from .event_refinement_dev import question_from
from .qa_diagnostics import report_digest
from .types import Video


def validate_args(args):
    if len(set(args.gpu_ids)) != 2 or min(args.gpu_ids) < 0:
        raise ValueError("Select two distinct nonnegative GPU indices")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Run `unset CUDA_VISIBLE_DEVICES` before this suite")
    if (
        not math.isfinite(args.minimum_free_gib)
        or args.minimum_free_gib <= 0
    ):
        raise ValueError("minimum-free-gib must be positive and finite")
    if len(set(args.budgets)) != len(args.budgets) or len(args.budgets) < 2:
        raise ValueError("Choose at least two distinct frame budgets")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("Repeated splits are not allowed")
    if args.dry_run and args.decode_check:
        raise ValueError("Choose --dry-run or --decode-check, not both")


def build_plan(args):
    """Hash every requested input before assigning work to either GPU."""
    validate_args(args)
    manifest_path = args.manifest.resolve()
    manifest = runner.read_json(manifest_path)
    requested = [
        v for v in manifest["videos"]
        if v["split"] in args.splits
        and (args.duration == "all" or v["duration"] == args.duration)
    ]
    missing = [
        str((manifest_path.parent / v["path"]).resolve())
        for v in requested
        if not (manifest_path.parent / v["path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} requested videos; no jobs launched. "
            "Prepare every selected video before running:\n"
            + "\n".join(missing[:20])
        )
    plan = {
        "version": 1,
        "experiment": "a4500_full_suite_v1",
        "config_path": str(args.config.resolve()),
        "manifest_path": str(manifest_path),
        "gpu_ids": list(args.gpu_ids),
        "budgets": sorted(args.budgets),
        "splits": {},
    }
    seen_content = {}
    for split in args.splits:
        inputs = SimpleNamespace(
            config=args.config, manifest=manifest_path,
            frame_budget=max(args.budgets), split=split,
            duration=args.duration, max_videos=None, max_questions=None,
            num_shards=1, shard_index=0,
        )
        _, videos, protocol = runner.load_inputs(inputs)
        for vid, identity in protocol["inputs"].items():
            digest = identity["sha256"]
            if digest in seen_content:
                raise ValueError(
                    "Duplicate video content across requested splits: "
                    f"{seen_content[digest]} and {split}/{vid}"
                )
            seen_content[digest] = f"{split}/{vid}"
        ids = sorted(v["video_id"] for v in videos)
        count = min(2, len(ids))
        plan["splits"][split] = {
            "protocol": protocol,
            "shards": [
                {
                    "index": index,
                    "gpu": args.gpu_ids[index],
                    "video_ids": ids[index::count],
                }
                for index in range(count)
            ],
        }
    return plan


def plan_summary(plan):
    manifest = runner.read_json(plan["manifest_path"])
    videos = {v["video_id"]: v for v in manifest["videos"]}
    splits = {}
    for split, entry in plan["splits"].items():
        protocol = entry["protocol"]
        cohort = protocol["cohort"]
        per_budget = sum(
            len(protocol["orders_by_option_count"][str(len(q["options"]))])
            * len(runner.CONDITIONS)
            for vid in cohort for q in videos[vid]["questions"]
        )
        splits[split] = {
            "n_videos": len(cohort),
            "n_questions": sum(map(len, cohort.values())),
            "duration": protocol["duration"],
            "qa_trials_per_budget": per_budget,
            "qa_trials_all_budgets": per_budget * len(plan["budgets"]),
            "gpu_shards": [
                {
                    "gpu": shard["gpu"],
                    "n_videos": len(shard["video_ids"]),
                    "n_questions": sum(
                        len(cohort[vid]) for vid in shard["video_ids"]
                    ),
                }
                for shard in entry["shards"]
            ],
        }
    return {
        "suite_plan_sha256": report_digest(plan),
        "budgets": plan["budgets"],
        "conditions": list(runner.CONDITIONS),
        "splits": splits,
        "total_qa_trials": sum(
            s["qa_trials_all_budgets"] for s in splits.values()
        ),
        "note": "All requested questions; dev/eval reported separately. "
        "One process per GPU; VRAM is not pooled. Input hashing does not "
        "prove candidate eligibility; --decode-check verifies it on CPU.",
    }


def verify_candidates(plan):
    """Optional CPU-only exact decode audit at the largest frame budget."""
    manifest = runner.read_json(plan["manifest_path"])
    videos = {v["video_id"]: v for v in manifest["videos"]}
    seen, result = {}, []
    for split, entry in plan["splits"].items():
        protocol = entry["protocol"]
        config = Config(**protocol["config"])
        for vid, identity in protocol["inputs"].items():
            item = videos[vid]
            try:
                frames, times, duration = decode_video(
                    Path(identity["path"]), config
                )
                video = Video(
                    vid, split, frames, times, duration,
                    tuple(question_from(q) for q in item["questions"]),
                    item["source_id"],
                )
                validate_video(video, config)
                digest = array_digest(frames, times)
                if digest in seen:
                    raise ValueError(
                        f"Same decoded candidates as {seen[digest]}"
                    )
                seen[digest] = f"{split}/{vid}"
                row = {
                    "split": split, "video_id": vid,
                    "candidate_count": len(times),
                    "candidate_sha256": digest,
                }
                result.append(row)
                print("Decode check: " + json.dumps(row), flush=True)
                del video, frames, times
            except Exception as error:
                raise RuntimeError(
                    f"B{config.frame_budget} input check failed: "
                    f"{split}/{vid}, {identity['path']}: {error}"
                ) from error
    return result


def worker_specs(plan, root):
    """Keep each video shard on the same physical GPU across resumes."""
    root = Path(root).resolve()
    specs = []
    # Expensive budgets first; each GPU has its own fixed, sequential queue.
    for budget in reversed(plan["budgets"]):
        for split, entry in plan["splits"].items():
            for shard in entry["shards"]:
                index, gpu = shard["index"], shard["gpu"]
                job_id = f"{split}_b{budget}_s{index}"
                output = job_directory(root, split, budget, index)
                specs.append({
                    "id": job_id, "split": split, "budget": budget,
                    "shard_index": index, "gpu": gpu,
                    "output": str(output),
                    "log": str(root / "logs" / f"{job_id}.log"),
                    "command": [
                        sys.executable, "-u", "-m",
                        "event_caption_pilot.portable_budget",
                        "--config", plan["config_path"],
                        "--manifest", plan["manifest_path"],
                        "--output-dir", str(output),
                        "--frame-budget", str(budget),
                        "--split", split,
                        "--duration", entry["protocol"]["duration"],
                        "--num-shards", str(len(entry["shards"])),
                        "--shard-index", str(index),
                        "--expected-protocol-sha256", report_digest(
                            expected_protocol(plan, split, budget, index)
                        ),
                    ],
                    "environment": {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONHASHSEED": "42",
                        "OMP_NUM_THREADS": "4",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    },
                })
    return specs


def execute_workers(specs, on_complete=None):
    """At most one child per GPU; stop on failure and preserve checkpoints."""
    queues = {}
    for spec in specs:
        queues.setdefault(spec["gpu"], deque()).append(spec)
    active, completed = {}, []
    try:
        while any(queues.values()) or active:
            for gpu, queue in queues.items():
                if queue and gpu not in active:
                    spec = queue.popleft()
                    path = Path(spec["log"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    log = path.open("a", encoding="utf-8")
                    try:
                        log.write("\nLaunching: " + json.dumps(spec) + "\n")
                        log.flush()
                        process = subprocess.Popen(
                            spec["command"],
                            env={**os.environ, **spec["environment"]},
                            stdout=log, stderr=subprocess.STDOUT,
                        )
                    except BaseException:
                        log.close()
                        raise
                    active[gpu] = (process, log, spec)
                    print(
                        f"Started {spec['id']} on GPU {gpu}; "
                        f"log: {path}", flush=True,
                    )
            for gpu, (process, log, spec) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code:
                    raise RuntimeError(
                        f"{spec['id']} on GPU {gpu} exited {code}; "
                        f"inspect {spec['log']}. Rerun the same command "
                        "to resume saved trials."
                    )
                completed.append(spec["id"])
                if on_complete is not None:
                    on_complete(list(completed))
                print(
                    f"Completed {spec['id']} ({len(completed)}/"
                    f"{len(specs)} jobs)", flush=True,
                )
            if active:
                time.sleep(0.5)
    finally:
        for process, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, log, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                log.close()
    return completed


def freeze_plan(root, plan):
    path = root / "suite_plan.json"
    if path.exists():
        if runner.read_json(path) != plan:
            raise ValueError(
                "Suite inputs/code/settings/GPU assignment changed; "
                "use the original command/environment or a new output root"
            )
    else:
        if (root / "jobs").exists() or (root / "comparison.json").exists():
            raise ValueError("Existing outputs lack a matching suite plan")
        runner.save_json(path, plan)


def run_suite(args, plan):
    root = args.output_root.resolve()
    with runner.output_lock(root):
        freeze_plan(root, plan)
        status_path = root / "suite_status.json"
        status = {
            "status": "running",
            "suite_plan_sha256": report_digest(plan),
            "completed_jobs": [],
        }
        runner.save_json(status_path, status)
        runner.save_json(root / "comparison.json", {
            "status": "running", "completed": False,
            "suite_plan_sha256": report_digest(plan),
        })

        def record_progress(completed):
            status["completed_jobs"] = completed
            runner.save_json(status_path, status)

        try:
            if args.decode_check:
                runner.save_json(root / "input_decode_check.json", {
                    "suite_plan_sha256": report_digest(plan),
                    "videos": verify_candidates(plan),
                })
            # CUDA preflight exits before children load model weights.
            subprocess.run([
                sys.executable, "-m", "event_caption_pilot.a4500_suite",
                "check", "--gpu-ids", *(str(g) for g in args.gpu_ids),
                "--minimum-free-gib", str(args.minimum_free_gib),
            ], env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                check=True)
            status["completed_jobs"] = execute_workers(
                worker_specs(plan, root), on_complete=record_progress
            )
            aggregate(root, plan)
            status["status"] = "complete"
            runner.save_json(status_path, status)
        except BaseException as error:
            status.update(status="failed", error=str(error))
            runner.save_json(status_path, status)
            runner.save_json(root / "comparison.json", {
                **status, "completed": False,
            })
            raise
    print(f"Complete: {root / 'comparison.json'}", flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("check", "run", "summarize"))
    result.add_argument(
        "--config", type=Path,
        default=Path("configs/real.a4500.pinned.json"),
    )
    result.add_argument(
        "--manifest", type=Path,
        default=Path("data/videomme_a4500/manifest.json"),
    )
    result.add_argument(
        "--output-root", type=Path, default=Path("outputs/a4500_full")
    )
    result.add_argument("--gpu-ids", nargs=2, type=int, default=[0, 1])
    result.add_argument("--minimum-free-gib", type=float, default=16.0)
    result.add_argument(
        "--budgets", nargs="+", type=int, choices=(16, 32, 64),
        default=[16, 32, 64],
    )
    result.add_argument(
        "--splits", nargs="+", choices=("dev", "eval"),
        default=["dev", "eval"],
    )
    result.add_argument(
        "--duration", choices=("all", "short", "medium", "long"),
        default="all",
    )
    result.add_argument(
        "--decode-check", action="store_true",
        help="Decode all selected videos on CPU to check exact eligibility",
    )
    result.add_argument(
        "--dry-run", action="store_true",
        help="Validate/hash inputs and print jobs without writes or CUDA",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == "summarize":
        if args.dry_run or args.decode_check:
            raise ValueError("summarize does not accept run/check flags")
        root = args.output_root.resolve()
        plan = runner.read_json(root / "suite_plan.json")
        with runner.output_lock(root):
            status = {
                "status": "validating",
                "suite_plan_sha256": report_digest(plan),
                "note": "Recomputed from validated saved trials; no GPU use",
            }
            runner.save_json(root / "suite_status.json", status)
            try:
                aggregate(root, plan)
                status["status"] = "complete"
                runner.save_json(root / "suite_status.json", status)
            except BaseException as error:
                status.update(status="failed", error=str(error))
                runner.save_json(root / "suite_status.json", status)
                runner.save_json(root / "comparison.json", {
                    **status, "completed": False,
                })
                raise
        print(f"Complete: {root / 'comparison.json'}", flush=True)
        return 0
    plan = build_plan(args)
    print(json.dumps(plan_summary(plan), indent=2), flush=True)
    if args.dry_run:
        print(json.dumps(worker_specs(plan, args.output_root), indent=2))
        return 0
    if args.stage == "check":
        if args.decode_check:
            verify_candidates(plan)
        return 0
    return run_suite(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
