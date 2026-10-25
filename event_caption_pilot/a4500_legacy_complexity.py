"""Frozen-candidate Legacy pixel complexity and allocation comparison."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from .a4500_full_suite import execute_workers
from .a4500_long_ablation import separate_output
from .aggregate_budget_shards import rooted, save_csv
from .legacy_complexity_analysis import (
    ALGORITHM_SPEC,
    CONDITIONS,
    make_conditions,
    pixel_complexity,
    representative_indices,
    validate_selection,
)
from .long_ablation_analysis import _resources
from .long_ablation_source import (
    code_hashes,
    validate_snapshot,
    verify_source_code,
)
from .long_ablation_worker import _config, _videos
from .paired_reanalysis import clustered_means
from .portable_budget import (
    checkpoint,
    file_identity,
    output_lock,
    read_json,
    save_json,
)
from .qa_diagnostics import report_digest
from .video_sampling import check_decoded_native_count, plan_native_candidates

EXPERIMENT = "a4500_legacy_complexity_v1"


def import_source(source):
    from .a4500_segment_count import read_frozen as read_source
    from .segment_count_worker import validate_suite_preflight, validate_worker

    source = Path(source).resolve()
    old_plan, baseline = read_source(
        source / "suite_plan.json", verify_code=False
    )
    # Additive modules are allowed; recorded inference code remains frozen.
    verify_source_code(old_plan["code_sha256"])
    for protocol in baseline["source_protocols"].values():
        verify_source_code(protocol["code_sha256"])
    video_plans = validate_suite_preflight(source, old_plan, baseline)
    fresh = {}
    for shard in old_plan["shards"]:
        for row in validate_worker(source, old_plan, baseline, shard["index"]):
            key = row["video_id"], row["question_id"]
            if key in fresh:
                raise ValueError("Duplicate source question")
            fresh[key] = row
    rows = []
    for old in baseline["baseline_rows"]:
        key = old["video_id"], old["question_id"]
        row = deepcopy(old)
        row["conditions"].update(deepcopy(fresh.pop(key)["conditions"]))
        rows.append(row)
    if fresh or len(rows) != 60 or len(video_plans) != 20:
        raise ValueError(
            "Require complete dev Long 20-video/60-question source"
        )
    return {
        "source_root": str(source),
        "source_plan": old_plan,
        "baseline": baseline,
        "source_rows": rows,
        "source_video_plans": {p["video_id"]: p for p in video_plans},
    }


def build_plan(snapshot, cap=16, gpu_ids=(0, 1)):
    baseline = snapshot["baseline"]
    plan = {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_root": snapshot["source_root"],
        "snapshot_sha256": report_digest(snapshot),
        "algorithm": deepcopy(ALGORITHM_SPEC),
        "cap": cap,
        "partition": f"event_k{cap}",
        "conditions": list(CONDITIONS),
        "shards": deepcopy(baseline["source_shards"]),
        "gpu_ids": list(gpu_ids),
        "n_videos": 20,
        "n_questions": 60,
        "expected_qa_trials": 60 * len(CONDITIONS) * 6,
        "frame_budget": 32,
        "qa_side": 320,
        "split": "dev",
        "duration": "long",
        "code_sha256": code_hashes(),
        "adaptation": (
            "Legacy scoring/allocation on frozen SigLIP2 D2 "
            "candidate-index partitions; original pixel detector, "
            "raw-frame pool, CLIP and Qwen2-VL are not replayed."
        ),
    }
    validate_frozen(plan, snapshot)
    return plan


def validate_frozen(plan, snapshot):
    baseline = snapshot["baseline"]
    validate_snapshot(baseline)
    if (
        plan.get("experiment") != EXPERIMENT
        or plan.get("snapshot_sha256") != report_digest(snapshot)
        or plan.get("algorithm") != ALGORITHM_SPEC
        or plan.get("cap") not in (16, 24, 32)
        or plan.get("partition") != f"event_k{plan['cap']}"
        or plan.get("conditions") != list(CONDITIONS)
        or plan.get("source_root") != snapshot["source_root"]
        or plan.get("shards") != baseline["source_shards"]
        or plan.get("gpu_ids") != [s["gpu"] for s in baseline["source_shards"]]
        or len(set(plan["gpu_ids"])) != 2
        or plan.get("frame_budget") != 32
        or plan.get("qa_side") != 320
        or plan.get("n_videos") != 20
        or plan.get("n_questions") != 60
        or plan.get("expected_qa_trials") != 60 * len(CONDITIONS) * 6
        or not plan.get("code_sha256")
    ):
        raise ValueError("Legacy comparison frozen plan mismatch")
    keys = [(r["video_id"], r["question_id"]) for r in snapshot["source_rows"]]
    expected = {(v, q) for v, qs in baseline["cohort"].items() for q in qs}
    if len(keys) != 60 or set(keys) != expected or len(set(keys)) != 60:
        raise ValueError("Legacy comparison source cohort mismatch")
    if set(snapshot["source_video_plans"]) != set(baseline["cohort"]):
        raise ValueError("Missing frozen source video plans")


def read_frozen(plan_path, expected_digest=None, verify_code=True):
    path = Path(plan_path).resolve()
    plan, snapshot = (
        read_json(path),
        read_json(path.parent / "source_snapshot.json"),
    )
    validate_frozen(plan, snapshot)
    if expected_digest is not None and report_digest(plan) != expected_digest:
        raise ValueError("Legacy plan digest changed")
    if verify_code:
        if plan["code_sha256"] != code_hashes():
            raise ValueError("Code changed after Legacy comparison planning")
        verify_source_code(snapshot["source_plan"]["code_sha256"])
        for protocol in snapshot["baseline"]["source_protocols"].values():
            verify_source_code(protocol["code_sha256"])
    return plan, snapshot


def decode_scoring_pixels(path, config, frozen, wanted):
    """Verify frozen candidates; retain native-to-Pillow64 samples."""
    import cv2
    from PIL import Image

    count = frozen["candidate_count"]
    digest = hashlib.sha256()
    digest.update(
        str(
            (
                (count, config.frame_height, config.frame_width, 3),
                np.dtype(np.uint8).str,
            )
        ).encode()
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {path}")
    times, pixels = [], {}
    frame_index, next_observation = 0, 0.0
    native, sampling = None, None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if config.min_candidate_frames > 0:
            planned, sampling = plan_native_candidates(
                config, fps, capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            native = set(planned)
            if len(planned) != count:
                raise ValueError("Candidate count changed")
        elif not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source FPS")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stamp = frame_index / fps
            selected = (
                frame_index in native
                if native is not None
                else stamp >= next_observation
            )
            if selected:
                index = len(times)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small = cv2.resize(
                    rgb, (config.frame_width, config.frame_height)
                )
                digest.update(memoryview(small).cast("B"))
                if index in wanted:
                    # Legacy uses Pillow's default RGB BICUBIC resize.
                    pixels[index] = (
                        np.asarray(
                            Image.fromarray(rgb).resize((64, 64)),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
                times.append(stamp)
                if native is None:
                    next_observation += 1.0 / min(config.candidate_fps, fps)
                if (
                    len(times) > count
                    or len(times) > config.max_candidate_frames
                ):
                    raise ValueError("Too many candidates")
            frame_index += 1
        if sampling is not None:
            check_decoded_native_count(
                config,
                fps,
                sampling["native_frame_count"],
                frame_index,
                len(times),
            )
    finally:
        capture.release()
    times = np.asarray(times, dtype=float)
    digest.update(str((times.shape, times.dtype.str)).encode())
    digest.update(memoryview(times).cast("B"))
    if (
        len(times) != count
        or digest.hexdigest() != frozen["candidate_sha256"]
        or not np.array_equal(times, frozen["candidate_timestamps"])
        or frame_index / fps != frozen["duration_seconds"]
        or set(pixels) != set(wanted)
    ):
        raise ValueError("Legacy decode differs from frozen candidates")
    return pixels, times


def prepared_path(root, vid):
    return rooted(root, "plans", report_digest(vid) + ".json")


def prepared_identity(plan, vid):
    return {"suite_plan_sha256": report_digest(plan), "video_id": vid}


def allocation_audit(selections):
    r, cr = selections["legacy_R_uniform"], selections["legacy_CR_uniform"]
    count = len(r["events"])
    return {
        "actual_events": count,
        "budget_after_minimum": 32 - count,
        "allocation_R": r["allocation"],
        "allocation_CR": cr["allocation"],
        "allocation_changed": r["allocation"] != cr["allocation"],
        "uniform_replaced_frames": len(
            set(r["selected_indices"]) - set(cr["selected_indices"])
        ),
        "topk_replaced_frames": len(
            set(selections["legacy_R_topk"]["selected_indices"])
            - set(selections["legacy_CR_topk"]["selected_indices"])
        ),
    }


def validate_prepared_video(saved, vid, plan, snapshot):
    source = snapshot["source_video_plans"][vid]
    for key in (
        "video_id",
        "candidate_sha256",
        "candidate_count",
        "candidate_timestamps",
        "duration_seconds",
    ):
        if saved.get(key) != source[key]:
            raise ValueError(f"Prepared Legacy video identity changed: {key}")
    qids = snapshot["baseline"]["cohort"][vid]
    if set(saved["questions"]) != set(qids):
        raise ValueError("Prepared question cohort changed")
    if set(saved["complexity_by_cap"]) != {"16", "24", "32"} or set(
        saved["selection_audit_by_question_cap"]
    ) != set(qids):
        raise ValueError("Incomplete all-cap Legacy diagnostics")
    for qid, selections in saved["questions"].items():
        if set(selections) != set(CONDITIONS):
            raise ValueError("Prepared conditions changed")
        reference = source["questions"][qid][f"event_k{plan['cap']}_D"]
        for name, selection in selections.items():
            validate_selection(selection, name, saved["candidate_timestamps"])
            if (
                selection["events"] != reference["events"]
                or selection["frame_scores_raw_cosine"]
                != reference["frame_scores_raw_cosine"]
            ):
                raise ValueError(
                    "Legacy boundaries or relevance candidates changed"
                )
        expected_audit = {}
        for cap in (16, 24, 32):
            reference = source["questions"][qid][f"event_k{cap}_D"]
            replay = make_conditions(
                reference["events"],
                reference["frame_scores_raw_cosine"],
                saved["candidate_timestamps"],
                saved["complexity_by_cap"][str(cap)],
            )
            expected_audit[str(cap)] = allocation_audit(replay)
            if cap == plan["cap"] and replay != selections:
                raise ValueError("Legacy shared pixel complexity changed")
        if saved["selection_audit_by_question_cap"][qid] != expected_audit:
            raise ValueError("Legacy all-cap allocation audit changed")
    return saved


def prepare_video(root, plan, snapshot, item):
    vid = item["video_id"]
    baseline = snapshot["baseline"]
    source = snapshot["source_video_plans"][vid]
    identity = baseline["inputs"][vid]
    current = file_identity(Path(identity["path"]))
    if any(
        current[k] != identity[k] for k in ("path", "size_bytes", "sha256")
    ):
        raise ValueError(f"Original video changed: {vid}")

    def compute():
        started = perf_counter()
        first = next(iter(source["questions"].values()))
        bounds = {
            cap: first[f"event_k{cap}_D"]["events"] for cap in (16, 24, 32)
        }
        wanted = {
            i
            for b in bounds.values()
            for group in representative_indices(b)
            for i in group
        }
        pixels, times = decode_scoring_pixels(
            Path(identity["path"]), _config(baseline), source, wanted
        )
        complexities = {
            cap: pixel_complexity(b, pixels) for cap, b in bounds.items()
        }
        questions, audit = {}, {}
        for qid, old in source["questions"].items():
            audit[qid] = {}
            for cap, b in bounds.items():
                selections = make_conditions(
                    b,
                    old[f"event_k{cap}_D"]["frame_scores_raw_cosine"],
                    times,
                    complexities[cap],
                )
                audit[qid][str(cap)] = allocation_audit(selections)
                if cap == plan["cap"]:
                    questions[qid] = selections
        return {
            **{
                k: source[k]
                for k in (
                    "video_id",
                    "source_id",
                    "candidate_sha256",
                    "candidate_count",
                    "candidate_timestamps",
                    "duration_seconds",
                )
            },
            "questions": questions,
            "complexity_by_cap": {str(k): v for k, v in complexities.items()},
            "selection_audit_by_question_cap": audit,
            "prepare_wall_seconds": perf_counter() - started,
            "pixel_resize": (
                "native RGB -> Pillow RGB resize 64x64 -> float32/255"
            ),
        }

    saved = checkpoint(
        prepared_path(root, vid), prepared_identity(plan, vid), compute
    )
    return validate_prepared_video(saved, vid, plan, snapshot)


def selection_report(plans, plan):
    caps = {}
    for cap in (16, 24, 32):
        rows = [
            q[str(cap)]
            for v in plans.values()
            for q in v["selection_audit_by_question_cap"].values()
        ]
        caps[str(cap)] = {
            "questions": len(rows),
            "complexity_changes_allocation_questions": sum(
                r["allocation_changed"] for r in rows
            ),
            "mean_uniform_replaced_frames": float(
                np.mean([r["uniform_replaced_frames"] for r in rows])
            ),
            "mean_topk_replaced_frames": float(
                np.mean([r["topk_replaced_frames"] for r in rows])
            ),
            "actual_event_counts": sorted({r["actual_events"] for r in rows}),
        }
    return {
        "status": "prepared",
        "suite_plan_sha256": report_digest(plan),
        "prepared_plan_sha256": report_digest(plans),
        "n_videos": len(plans),
        "qa_executed_by_prepare": False,
        "caps": caps,
    }


def load_prepared(root, plan, snapshot):
    plans = {}
    for vid in sorted(snapshot["baseline"]["cohort"]):
        saved = checkpoint(
            prepared_path(root, vid), prepared_identity(plan, vid)
        )
        plans[vid] = validate_prepared_video(saved, vid, plan, snapshot)
    if read_json(rooted(root, "selection_preflight.json")) != selection_report(
        plans, plan
    ):
        raise ValueError(
            "Both-shard Legacy preparation is incomplete or changed"
        )
    return plans


def prepare(root, plan, snapshot):
    plans = {}
    # CPU-only: two decoders, no model construction or CUDA allocation.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(prepare_video, root, plan, snapshot, item)
            for shard in plan["shards"]
            for item in _videos(snapshot["baseline"], shard)
        ]
        for future in as_completed(futures):
            saved = future.result()
            plans[saved["video_id"]] = saved
            save_json(
                rooted(root, "prepare_progress.json"),
                {
                    "status": "preparing",
                    "prepared_videos": len(plans),
                    "total_videos": 20,
                },
            )
            print(
                f"Prepared Legacy pixels {len(plans)}/20: {saved['video_id']}",
                flush=True,
            )
    plans = dict(sorted(plans.items()))
    report = selection_report(plans, plan)
    save_json(rooted(root, "selection_preflight.json"), report)
    save_json(
        rooted(root, "prepare_progress.json"),
        {"status": "prepared", "prepared_videos": 20, "total_videos": 20},
    )
    return report


def worker_specs(root, plan, snapshot):
    from .legacy_complexity_worker import job_directory

    settings = snapshot["baseline"]["config"]
    return [
        {
            "id": job_directory(root, s["index"]).name,
            "gpu": s["gpu"],
            "output": str(job_directory(root, s["index"])),
            "log": str(rooted(root, "logs", f"legacy_s{s['index']}.log")),
            "command": [
                sys.executable,
                "-u",
                "-m",
                "event_caption_pilot.legacy_complexity_worker",
                "--plan",
                str(rooted(root, "suite_plan.json")),
                "--shard-index",
                str(s["index"]),
                "--expected-plan-sha256",
                report_digest(plan),
            ],
            "environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(s["gpu"]),
                "PYTHONHASHSEED": str(settings["seed"]),
                "OMP_NUM_THREADS": "4",
                "CUBLAS_WORKSPACE_CONFIG": settings["cublas_workspace_config"],
            },
        }
        for s in plan["shards"]
    ]


def aggregate(root):
    from .legacy_complexity_worker import validate_worker

    plan, snapshot = read_frozen(
        rooted(root, "suite_plan.json"), verify_code=False
    )
    fresh = [
        r
        for shard in plan["shards"]
        for r in validate_worker(root, plan, snapshot, shard["index"])
    ]
    indexed = {(r["video_id"], r["question_id"]): r for r in fresh}
    source = {
        (r["video_id"], r["question_id"]): r for r in snapshot["source_rows"]
    }
    if len(fresh) != 60 or len(indexed) != 60 or set(indexed) != set(source):
        raise ValueError("Legacy results lack the complete paired cohort")
    refs = [
        "uniform",
        "frame_top",
        "temporal_bin",
        "D",
        f"event_k{plan['cap']}_D",
        f"event_k{plan['cap']}_D_topk",
        "fixed_k32_D",
        "event_k32_D_topk",
    ]
    refs = list(dict.fromkeys(refs))
    contrasts = {
        "complexity_effect_uniform": ("legacy_CR_uniform", "legacy_R_uniform"),
        "complexity_effect_topk": ("legacy_CR_topk", "legacy_R_topk"),
        "within_selection_effect_R": ("legacy_R_topk", "legacy_R_uniform"),
        "within_selection_effect_CR": ("legacy_CR_topk", "legacy_CR_uniform"),
        **{
            f"{arm}_minus_reference_{ref}": (arm, ref)
            for arm in CONDITIONS
            for ref in refs
        },
    }
    metrics, resources = [], {}
    for key in sorted(source):
        row = {"video_id": key[0], "question_id": key[1]}
        for name in [*refs, *CONDITIONS]:
            value = indexed[key] if name in CONDITIONS else source[key]
            row[name] = (
                sum(
                    t["scoring_correct"]
                    for t in value["conditions"][name]["trials"]
                )
                / 6
            )
        row.update({k: row[a] - row[b] for k, (a, b) in contrasts.items()})
        metrics.append(row)
    for arm in CONDITIONS:
        pairs = [
            (t, p)
            for row in fresh
            for t, p in zip(
                row["conditions"][arm]["trials"],
                row["conditions"][arm]["trial_provenance"],
            )
        ]
        resources[arm] = {
            "trials": len(pairs),
            "new_trials": sum(p["kind"] == "new" for _, p in pairs),
            "reused_trials": sum(p["kind"] != "new" for _, p in pairs),
            "recorded_qa": _resources([t for t, _ in pairs], arm),
            "newly_computed_qa": _resources(
                [t for t, p in pairs if p["kind"] == "new"], arm
            ),
        }
    result = {
        "status": "complete",
        "completed": True,
        "experiment": EXPERIMENT,
        "suite_plan_sha256": report_digest(plan),
        "algorithm": ALGORITHM_SPEC,
        "adaptation": plan["adaptation"],
        "cap": plan["cap"],
        "frame_budget": 32,
        "n_videos": 20,
        "n_questions": 60,
        "conditions": list(CONDITIONS),
        "historical_references": refs,
        "metric_definitions": {
            k: f"{a} minus {b}" for k, (a, b) in contrasts.items()
        },
        "overall": clustered_means(
            metrics,
            [*refs, *CONDITIONS, *contrasts],
            seed=snapshot["baseline"]["config"]["seed"],
            samples=5000,
        ),
        "qa_resources": resources,
        "selection_preflight": read_json(
            rooted(root, "selection_preflight.json")
        ),
        "question_metrics": metrics,
        "questions": fresh,
        "interpretation": (
            "Exploratory dev Long; six repeated option orders, "
            "5000 video-cluster resamples, marginal 95% CIs "
            "without multiplicity correction. Historical "
            "comparisons bundle relevance, normalization, "
            "allocator and representative-selection changes. "
            "CR minus R isolates pixel complexity within the "
            "adapted Legacy policy. No full pipeline replication."
        ),
    }
    save_json(rooted(root, "comparison.json"), result)
    save_json(
        rooted(root, "handoff.json"),
        {k: v for k, v in result.items() if k != "questions"},
    )
    save_csv(rooted(root, "question_accuracy.csv"), list(metrics[0]), metrics)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("check", "prepare", "run", "summarize")
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/a4500_siglip2_segment_count"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/a4500_legacy_complexity_k16"),
    )
    parser.add_argument("--cap", type=int, choices=(16, 24, 32), default=16)
    parser.add_argument("--gpu-ids", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--minimum-free-gib", type=float, default=16.0)
    args = parser.parse_args(argv)
    if args.stage == "run" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError(
            "Unset parent CUDA_VISIBLE_DEVICES before two-GPU run"
        )
    if not np.isfinite(args.minimum_free_gib) or args.minimum_free_gib <= 0:
        raise ValueError("GPU free-memory requirement must be positive")
    root = args.output_root.resolve()
    if args.stage == "summarize":
        result = aggregate(root)
        print(result["overall"])
        return 0
    separate_output(root, args.source_root)
    path = root / "suite_plan.json"
    if path.exists():
        plan, snapshot = read_frozen(path)
        if (
            plan["cap"] != args.cap
            or plan["gpu_ids"] != args.gpu_ids
            or plan["source_root"] != str(args.source_root.resolve())
        ):
            raise ValueError(
                "Resume with the original source/cap/GPU arguments"
            )
    else:
        snapshot = import_source(args.source_root)
        plan = build_plan(snapshot, args.cap, args.gpu_ids)
    if args.stage == "check":
        print(
            {
                k: plan[k]
                for k in (
                    "cap",
                    "n_videos",
                    "n_questions",
                    "expected_qa_trials",
                    "gpu_ids",
                )
            }
        )
        return 0
    with output_lock(root):
        if not path.exists():
            if any(
                p.name not in {".run.lock", "source_snapshot.json"}
                for p in root.iterdir()
            ):
                raise ValueError("Use a new output directory")
            save_json(root / "source_snapshot.json", snapshot)
            save_json(path, plan)
        status = {
            "status": "preparing",
            "completed": False,
            "suite_plan_sha256": report_digest(plan),
        }
        save_json(root / "suite_status.json", status)
        try:
            preflight = prepare(root, plan, snapshot)
            print(preflight["caps"], flush=True)
            if args.stage == "prepare":
                prior = root / "comparison.json"
                complete = (
                    prior.exists()
                    and read_json(prior).get("completed") is True
                )
                if complete:
                    aggregate(root)
                status.update(
                    status="complete" if complete else "prepared",
                    completed=complete,
                )
            else:
                # Check both GPUs before constructing either model.
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "event_caption_pilot.a4500_suite",
                        "check",
                        "--gpu-ids",
                        *map(str, args.gpu_ids),
                        "--minimum-free-gib",
                        str(args.minimum_free_gib),
                    ],
                    env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                    check=True,
                )
                status.update(status="running", stage="qa")
                save_json(root / "suite_status.json", status)

                def progress(done):
                    status["completed_jobs"] = done
                    save_json(root / "suite_status.json", status)

                execute_workers(
                    worker_specs(root, plan, snapshot), on_complete=progress
                )
                aggregate(root)
                status.update(status="complete", completed=True)
        except BaseException as error:
            status.update(status="failed", error=str(error), completed=False)
            save_json(root / "suite_status.json", status)
            raise
        save_json(root / "suite_status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
