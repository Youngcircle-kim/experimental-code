"""Frozen A/B/C query-conditioned diversity experiment on two GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .a4500_full_suite import execute_workers
from .a4500_long_ablation import separate_output
from .aggregate_budget_shards import rooted, save_csv
from .algorithms import normalize_rows
from .backends import build_backend
from .complexity_ablation_analysis import _feature_digest
from .data import array_digest, decode_video
from .long_ablation_analysis import _resources
from .long_ablation_source import (
    code_hashes,
    validate_snapshot,
    verify_source_code,
)
from .long_ablation_worker import (
    _baseline_rows,
    _config,
    _shard,
    _validate_backend_identity,
    _validate_runtime,
    _videos,
)
from .order_experiments import seed_call
from .paired_reanalysis import clustered_means
from .portable_budget import (
    checkpoint,
    file_identity,
    output_lock,
    read_json,
    runtime_metadata,
    save_json,
)
from .qa_diagnostics import report_digest
from .query_diversity_analysis import (
    ALGORITHM_SPEC,
    CONDITIONS,
    make_conditions,
)
from .resolution_dev import decode_selected

EXPERIMENT = "a4500_query_diversity_abc_v1"
STRENGTH = 0.5
TEMPERATURE = 0.07


def now():
    return datetime.now(timezone.utc).isoformat()


def import_source(source):
    from .a4500_legacy_complexity import load_prepared as load_legacy
    from .a4500_legacy_complexity import read_frozen as read_legacy
    from .legacy_complexity_worker import validate_worker

    source = Path(source).resolve()
    old_plan, old = read_legacy(source / "suite_plan.json", verify_code=False)
    if old_plan["cap"] != 16:
        raise ValueError("Require the completed Legacy cap16 source")
    verify_source_code(old_plan["code_sha256"])
    verify_source_code(old["source_plan"]["code_sha256"])
    for protocol in old["baseline"]["source_protocols"].values():
        verify_source_code(protocol["code_sha256"])
    status = read_json(source / "suite_status.json")
    if status.get("status") != "complete" or not status.get("completed"):
        raise ValueError("Legacy source is not complete")
    print("Auditing completed Legacy source and checkpoints", flush=True)
    prepared = load_legacy(source, old_plan, old)
    rows = [
        row
        for shard in old_plan["shards"]
        for row in validate_worker(source, old_plan, old, shard["index"])
    ]
    hashes = {}
    for vid, saved in old["source_video_plans"].items():
        values = {
            q["event_k16_D"]["source_features_sha256"]
            for q in saved["questions"].values()
        }
        if len(values) != 1:
            raise ValueError("Source questions disagree on visual features")
        hashes[vid] = values.pop()
    return {
        "source_root": str(source),
        "source_plan": old_plan,
        "baseline": old["baseline"],
        "source_rows": rows,
        "source_video_plans": prepared,
        "source_feature_hashes": hashes,
    }


def build_plan(snapshot, gpu_ids=(0, 1)):
    plan = {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_root": snapshot["source_root"],
        "snapshot_sha256": report_digest(snapshot),
        "algorithm": deepcopy(ALGORITHM_SPEC),
        "conditions": list(CONDITIONS),
        "diversity_strength": STRENGTH,
        "diversity_temperature": TEMPERATURE,
        "cap": 16,
        "partition": "event_k16",
        "shards": deepcopy(snapshot["source_plan"]["shards"]),
        "gpu_ids": list(gpu_ids),
        "n_videos": 20,
        "n_questions": 60,
        "expected_qa_trials": 1080,
        "expected_A_source_reused_trials": 360,
        "maximum_new_qa_trials": 720,
        "frame_budget": 32,
        "qa_side": 320,
        "split": "dev",
        "duration": "long",
        "code_sha256": code_hashes(),
        "feature_cache": "original normalized float64; exact historical hash",
    }
    validate_frozen(plan, snapshot)
    return plan


def validate_frozen(plan, snapshot):
    baseline = snapshot["baseline"]
    validate_snapshot(baseline)
    expected = {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_root": snapshot["source_root"],
        "snapshot_sha256": report_digest(snapshot),
        "algorithm": ALGORITHM_SPEC,
        "conditions": list(CONDITIONS),
        "diversity_strength": STRENGTH,
        "diversity_temperature": TEMPERATURE,
        "cap": 16,
        "partition": "event_k16",
        "frame_budget": 32,
        "qa_side": 320,
        "split": "dev",
        "duration": "long",
        "n_videos": 20,
        "n_questions": 60,
        "expected_qa_trials": 1080,
        "expected_A_source_reused_trials": 360,
        "maximum_new_qa_trials": 720,
        "shards": snapshot["source_plan"]["shards"],
        "gpu_ids": [s["gpu"] for s in baseline["source_shards"]],
        "feature_cache": "original normalized float64; exact historical hash",
    }
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("A/B/C frozen plan changed")
    keys = {(v, q) for v, qs in baseline["cohort"].items() for q in qs}
    rows = snapshot["source_rows"]
    if (
        len(keys) != 60
        or len(baseline["cohort"]) != 20
        or len(rows) != 60
        or {(r["video_id"], r["question_id"]) for r in rows} != keys
        or set(snapshot["source_video_plans"]) != set(baseline["cohort"])
        or set(snapshot["source_feature_hashes"]) != set(baseline["cohort"])
        or len(set(plan["gpu_ids"])) != 2
        or not plan.get("code_sha256")
    ):
        raise ValueError("A/B/C source cohort or GPU assignment changed")
    for row in rows:
        vid, qid = row["video_id"], row["question_id"]
        ref = snapshot["source_video_plans"][vid]["questions"][qid][
            "legacy_R_topk"
        ]
        if any(
            row["conditions"]["legacy_R_topk"].get(k) != v
            for k, v in ref.items()
        ):
            raise ValueError("Source A selection differs from its plan")


def read_frozen(plan_path, expected_digest=None, verify_code=True):
    path = Path(plan_path).resolve()
    plan = read_json(path)
    snapshot = read_json(path.parent / "source_snapshot.json")
    validate_frozen(plan, snapshot)
    if expected_digest is not None and report_digest(plan) != expected_digest:
        raise ValueError("A/B/C plan digest changed")
    if verify_code:
        if plan["code_sha256"] != code_hashes():
            raise ValueError("Code changed after A/B/C planning")
        verify_source_code(snapshot["source_plan"]["code_sha256"])
        for p in snapshot["baseline"]["source_protocols"].values():
            verify_source_code(p["code_sha256"])
    return plan, snapshot


def prepared_path(root, vid):
    return rooted(root, "plans", report_digest(vid) + ".json")


def prepared_identity(plan, vid):
    return {"suite_plan_sha256": report_digest(plan), "video_id": vid}


def feature_path(root, vid):
    return rooted(root, "features", report_digest(vid) + ".npy")


def save_features(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, features, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_features(root, vid, saved, snapshot):
    features = np.load(feature_path(root, vid), allow_pickle=False)
    if (
        features.dtype != np.float64
        or features.ndim != 2
        or features.shape[0] != saved["candidate_count"]
        or list(features.shape) != saved["feature_shape"]
        or not np.isfinite(features).all()
        or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-12)
        or array_digest(features) != saved["features_sha256"]
        or _feature_digest(features) != snapshot["source_feature_hashes"][vid]
        or saved["source_features_sha256"]
        != snapshot["source_feature_hashes"][vid]
    ):
        raise ValueError("Candidate embedding cache differs from source")
    return features


def validate_prepared_video(root, saved, vid, plan, snapshot):
    source = snapshot["source_video_plans"][vid]
    for key in (
        "video_id",
        "source_id",
        "candidate_sha256",
        "candidate_count",
        "candidate_timestamps",
        "duration_seconds",
    ):
        if saved.get(key) != source[key]:
            raise ValueError(f"A/B/C candidate identity changed: {key}")
    if set(saved["questions"]) != set(snapshot["baseline"]["cohort"][vid]):
        raise ValueError("A/B/C prepared question cohort changed")
    features = load_features(root, vid, saved, snapshot)
    sources = {
        r["question_id"]: r
        for r in snapshot["source_rows"]
        if r["video_id"] == vid
    }
    for qid, selections in saved["questions"].items():
        reference = source["questions"][qid]["legacy_R_topk"]
        expected = make_conditions(
            reference,
            features,
            saved["candidate_timestamps"],
            strength=plan["diversity_strength"],
            temperature=plan["diversity_temperature"],
        )
        if expected != selections:
            raise ValueError("A/B/C selection does not reproduce from cache")
        if (
            saved["verified_A_qa_pixels_sha256"][qid]
            != sources[qid]["conditions"]["legacy_R_topk"]["qa_pixels_sha256"]
        ):
            raise ValueError("A source QA input pixels changed")
    return saved


def prepare_video(root, plan, snapshot, item, get_model):
    vid = item["video_id"]
    baseline = snapshot["baseline"]
    config = _config(baseline)
    source = snapshot["source_video_plans"][vid]
    identity = baseline["inputs"][vid]
    current = file_identity(Path(identity["path"]))
    if any(
        current[k] != identity[k] for k in ("path", "size_bytes", "sha256")
    ):
        raise ValueError(f"Source video changed: {vid}")

    def compute():
        started = perf_counter()
        frames, times, duration = decode_video(Path(identity["path"]), config)
        decode_seconds = perf_counter() - started
        if (
            array_digest(frames, times) != source["candidate_sha256"]
            or times.tolist() != source["candidate_timestamps"]
            or duration != source["duration_seconds"]
        ):
            raise ValueError("Decoded A/B/C candidates differ from source")
        model = get_model()
        encoding_started = perf_counter()
        features = normalize_rows(
            model.encode_frames(frames), config.normalization_epsilon
        )
        encode_seconds = perf_counter() - encoding_started
        del frames
        if _feature_digest(features) != snapshot["source_feature_hashes"][vid]:
            raise ValueError("Historical SigLIP2 features do not reproduce")
        save_features(feature_path(root, vid), features)
        selection_started = perf_counter()
        questions = {
            qid: make_conditions(
                old["legacy_R_topk"],
                features,
                times,
                strength=plan["diversity_strength"],
                temperature=plan["diversity_temperature"],
            )
            for qid, old in source["questions"].items()
        }
        selection_seconds = perf_counter() - selection_started
        baselines = _baseline_rows(baseline)
        sources = {
            r["question_id"]: r
            for r in snapshot["source_rows"]
            if r["video_id"] == vid
        }
        union = sorted(
            {
                i
                for qid, selected in questions.items()
                for arm in [
                    selected[CONDITIONS[0]],
                    baselines[(vid, qid)]["conditions"]["D"],
                ]
                for i in arm["selected_indices"]
            }
        )
        low, pixels, replay_times, replay_duration, _ = decode_selected(
            Path(identity["path"]),
            config,
            len(times),
            union,
            320,
            source["candidate_sha256"],
        )
        del low
        if (
            not np.array_equal(times, replay_times)
            or duration != replay_duration
        ):
            raise ValueError("A source QA replay candidate grid changed")
        verified = {}
        for qid, selected in questions.items():
            for arm, old in (
                (
                    selected[CONDITIONS[0]],
                    sources[qid]["conditions"]["legacy_R_topk"],
                ),
                (
                    baselines[(vid, qid)]["conditions"]["D"],
                    baselines[(vid, qid)]["conditions"]["D"],
                ),
            ):
                indices = arm["selected_indices"]
                digest = array_digest(
                    np.stack([pixels[i] for i in indices]), times[indices]
                )
                if digest != old["qa_pixels_sha256"]:
                    raise ValueError("Source QA320 pixel hash changed")
            indices = selected[CONDITIONS[0]]["selected_indices"]
            verified[qid] = array_digest(
                np.stack([pixels[i] for i in indices]), times[indices]
            )
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
            "features_sha256": array_digest(features),
            "source_features_sha256": _feature_digest(features),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "verified_A_qa_pixels_sha256": verified,
            "decode_seconds": decode_seconds,
            "encode_seconds": encode_seconds,
            "selection_seconds": selection_seconds,
            "prepare_wall_seconds": perf_counter() - started,
        }

    saved = checkpoint(
        prepared_path(root, vid), prepared_identity(plan, vid), compute
    )
    return validate_prepared_video(root, saved, vid, plan, snapshot)


def prepare_shard(args):
    path = args.plan.resolve()
    root = path.parent
    plan, snapshot = read_frozen(path, args.expected_plan_sha256)
    shard = _shard(plan, args.shard_index)
    config = _config(snapshot["baseline"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(shard["gpu"]):
        raise ValueError("Feature preparation GPU differs from source")
    output = rooted(root, "preparation", f"s{shard['index']}")
    with output_lock(output):
        runtime = runtime_metadata()
        _validate_runtime(runtime, snapshot["baseline"], shard)
        runtime_path = output / "runtime.json"
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError("Feature preparation runtime changed")
        save_json(runtime_path, runtime)
        model = None

        def get_model():
            nonlocal model
            if model is None:
                seed_call(config)
                model = build_backend(config)
                metadata = model.metadata()
                _validate_backend_identity(
                    metadata, snapshot["baseline"], config
                )
                prior = output / "backend.json"
                if prior.exists() and read_json(prior) != metadata:
                    raise ValueError("Feature preparation backend changed")
                save_json(prior, metadata)
            return model

        done = []
        progress = {
            "status": "preparing",
            "suite_plan_sha256": report_digest(plan),
            "prepared_videos": 0,
            "total_videos": len(shard["video_ids"]),
        }
        save_json(output / "progress.json", progress)
        try:
            for item in _videos(snapshot["baseline"], shard):
                prepare_video(root, plan, snapshot, item, get_model)
                done.append(item["video_id"])
                progress.update(prepared_videos=len(done), video_ids=done)
                save_json(output / "progress.json", progress)
                print(
                    f"Prepared A/B/C embeddings {len(done)}/"
                    f"{len(shard['video_ids'])}: {item['video_id']}",
                    flush=True,
                )
            progress["status"] = "complete"
            save_json(output / "progress.json", progress)
        except BaseException as error:
            progress.update(
                status="failed", error=f"{type(error).__name__}: {error}"
            )
            save_json(output / "progress.json", progress)
            raise


def selection_report(prepared, plan):
    rows = [q for p in prepared.values() for q in p["questions"].values()]
    contrasts = {}
    for label, first, second in (
        ("B_minus_A", CONDITIONS[1], CONDITIONS[0]),
        ("C_minus_A", CONDITIONS[2], CONDITIONS[0]),
        ("C_minus_B", CONDITIONS[2], CONDITIONS[1]),
    ):
        contrasts[label] = {
            "allocation_changed_questions": sum(
                q[first]["allocation"] != q[second]["allocation"] for q in rows
            ),
            "selection_changed_questions": sum(
                q[first]["selected_indices"] != q[second]["selected_indices"]
                for q in rows
            ),
            "mean_replaced_frames": float(
                np.mean(
                    [
                        len(
                            set(q[first]["selected_indices"])
                            - set(q[second]["selected_indices"])
                        )
                        for q in rows
                    ]
                )
            ),
        }
    return {
        "status": "prepared",
        "suite_plan_sha256": report_digest(plan),
        "prepared_plan_sha256": report_digest(prepared),
        "n_videos": len(prepared),
        "n_questions": len(rows),
        "qa_executed_by_prepare": False,
        "comparisons": contrasts,
        "selection_diagnostics": {
            arm: {
                key: float(np.mean([q[arm]["diagnostics"][key] for q in rows]))
                for key in (
                    "selected_feature_redundancy",
                    "selected_mean_question_cosine",
                )
            }
            for arm in CONDITIONS
        },
        "recorded_preparation_seconds_sum": {
            key: sum(p[key] for p in prepared.values())
            for key in (
                "decode_seconds",
                "encode_seconds",
                "selection_seconds",
                "prepare_wall_seconds",
            )
        },
    }


def prepared_records(root, plan, snapshot):
    return {
        vid: validate_prepared_video(
            root,
            checkpoint(prepared_path(root, vid), prepared_identity(plan, vid)),
            vid,
            plan,
            snapshot,
        )
        for vid in sorted(snapshot["baseline"]["cohort"])
    }


def load_prepared(root, plan, snapshot):
    prepared = prepared_records(root, plan, snapshot)
    for shard in plan["shards"]:
        output = rooted(root, "preparation", f"s{shard['index']}")
        progress = read_json(output / "progress.json")
        if (
            progress.get("status") != "complete"
            or progress.get("suite_plan_sha256") != report_digest(plan)
            or progress.get("video_ids") != shard["video_ids"]
            or progress.get("prepared_videos") != len(shard["video_ids"])
        ):
            raise ValueError("Feature preparation shard incomplete")
        _validate_runtime(
            read_json(output / "runtime.json"), snapshot["baseline"], shard
        )
        _validate_backend_identity(
            read_json(output / "backend.json"),
            snapshot["baseline"],
            _config(snapshot["baseline"]),
        )
    if read_json(rooted(root, "selection_preflight.json")) != selection_report(
        prepared, plan
    ):
        raise ValueError("A/B/C preflight report changed")
    return prepared


def worker_specs(root, plan, snapshot, prepare=False):
    from .query_diversity_worker import job_directory

    settings = snapshot["baseline"]["config"]
    specs = []
    for shard in plan["shards"]:
        index = shard["index"]
        output = (
            rooted(root, "preparation", f"s{index}")
            if prepare
            else (job_directory(root, index))
        )
        specs.append(
            {
                "id": f"{'prepare' if prepare else 'qa'}_abc_s{index}",
                "gpu": shard["gpu"],
                "output": str(output),
                "log": str(
                    rooted(
                        root,
                        "logs",
                        f"{'prepare' if prepare else 'qa'}_s{index}.log",
                    )
                ),
                "command": [
                    sys.executable,
                    "-u",
                    "-m",
                    "event_caption_pilot.a4500_query_diversity"
                    if prepare
                    else "event_caption_pilot.query_diversity_worker",
                    *(["prepare-shard"] if prepare else []),
                    "--plan",
                    str(rooted(root, "suite_plan.json")),
                    "--shard-index",
                    str(index),
                    "--expected-plan-sha256",
                    report_digest(plan),
                ],
                "environment": {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(shard["gpu"]),
                    "PYTHONHASHSEED": str(settings["seed"]),
                    "OMP_NUM_THREADS": "4",
                    "CUBLAS_WORKSPACE_CONFIG": settings[
                        "cublas_workspace_config"
                    ],
                },
            }
        )
    return specs


def aggregate(root):
    from .query_diversity_worker import validate_worker

    plan, snapshot = read_frozen(rooted(root, "suite_plan.json"))
    rows = [
        r
        for s in plan["shards"]
        for r in validate_worker(root, plan, snapshot, s["index"])
    ]
    indexed = {(r["video_id"], r["question_id"]): r for r in rows}
    expected = {
        (r["video_id"], r["question_id"]) for r in snapshot["source_rows"]
    }
    if len(rows) != 60 or len(indexed) != 60 or set(indexed) != expected:
        raise ValueError("Incomplete A/B/C paired cohort")
    contrasts = {
        "B_minus_A": (CONDITIONS[1], CONDITIONS[0]),
        "C_minus_A": (CONDITIONS[2], CONDITIONS[0]),
        "C_minus_B": (CONDITIONS[2], CONDITIONS[1]),
    }
    metrics, resources = [], {}
    for key in sorted(indexed):
        q = indexed[key]
        value = {"video_id": key[0], "question_id": key[1]}
        for arm in CONDITIONS:
            value[arm] = (
                sum(
                    t["scoring_correct"]
                    for t in q["conditions"][arm]["trials"]
                )
                / 6
            )
        value.update(
            {k: value[a] - value[b] for k, (a, b) in contrasts.items()}
        )
        metrics.append(value)
    for arm in CONDITIONS:
        pairs = [
            (t, p)
            for q in rows
            for t, p in zip(
                q["conditions"][arm]["trials"],
                q["conditions"][arm]["trial_provenance"],
                strict=True,
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
        resources[arm]["recorded_qa"].update(
            historical_baseline=arm == CONDITIONS[0],
            timing_origin=(
                "frozen_legacy_R_topk_source_trials"
                if arm == CONDITIONS[0]
                else "recorded_trials_including_source_or_current_reuse"
            ),
        )
        resources[arm]["newly_computed_qa"].update(
            historical_baseline=False,
            timing_origin="new_QA_only_excludes_preparation_and_reused_trials",
        )
    if resources[CONDITIONS[0]]["new_trials"] != 0:
        raise ValueError("A must reuse all 360 frozen source trials")
    result = {
        "status": "complete",
        "completed": True,
        "experiment": EXPERIMENT,
        "suite_plan_sha256": report_digest(plan),
        "algorithm": ALGORITHM_SPEC,
        "diversity_strength": STRENGTH,
        "diversity_temperature": TEMPERATURE,
        "n_videos": 20,
        "n_questions": 60,
        "frame_budget": 32,
        "cap": 16,
        "conditions": list(CONDITIONS),
        "metric_definitions": {
            k: f"{a} minus {b}" for k, (a, b) in contrasts.items()
        },
        "primary_contrast": "C_minus_B",
        "overall": clustered_means(
            metrics,
            [*CONDITIONS, *contrasts],
            seed=snapshot["baseline"]["config"]["seed"],
            samples=5000,
        ),
        "qa_resources": resources,
        "selection_preflight": read_json(
            rooted(root, "selection_preflight.json")
        ),
        "question_metrics": metrics,
        "questions": rows,
        "interpretation": (
            "Exploratory dev Long, not held-out confirmation. Six option "
            "orders are repeats; independent uncertainty clusters are 20 "
            "videos. 5000 paired video bootstrap resamples; marginal 95% "
            "CIs without multiplicity correction. C minus B isolates query "
            "weighting of diversity, B minus A adds uniform feature "
            "diversity, C minus A is total added-score effect. Frozen "
            "lambda=.5 and tau=.07; no label-based tuning. Diversity is "
            "a proxy, not measured answer information or evidence recall."
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
        "stage",
        choices=(
            "check",
            "prepare",
            "run",
            "summarize",
            "prepare-shard",
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/a4500_legacy_complexity_k16"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/a4500_query_diversity_abc"),
    )
    parser.add_argument("--gpu-ids", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.stage == "prepare-shard":
        if (
            args.plan is None
            or args.shard_index is None
            or not args.expected_plan_sha256
        ):
            parser.error("prepare-shard requires plan, shard and digest")
        prepare_shard(args)
        return 0
    root = args.output_root.resolve()
    if args.stage == "summarize":
        print(aggregate(root)["overall"])
        return 0
    if args.stage in ("run", "prepare") and os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise ValueError("Unset parent CUDA_VISIBLE_DEVICES for two GPUs")
    separate_output(root, args.source_root)
    started, started_at = perf_counter(), now()
    path = root / "suite_plan.json"
    if path.exists():
        plan, snapshot = read_frozen(path)
        if (
            plan["source_root"] != str(args.source_root.resolve())
            or plan["gpu_ids"] != args.gpu_ids
        ):
            raise ValueError("Resume with original source and GPU IDs")
    else:
        snapshot = import_source(args.source_root)
        plan = build_plan(snapshot, args.gpu_ids)
    if args.stage == "check":
        print(
            {
                k: plan[k]
                for k in (
                    "conditions",
                    "n_videos",
                    "n_questions",
                    "maximum_new_qa_trials",
                    "gpu_ids",
                    "diversity_strength",
                    "diversity_temperature",
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
                raise ValueError("Choose an empty new output directory")
            save_json(root / "source_snapshot.json", snapshot)
            save_json(path, plan)
        prior = root / "suite_status.json"
        if prior.exists() and read_json(prior).get("completed") is True:
            aggregate(root)
            print("A/B/C already complete and verified")
            return 0
        status = {
            "status": "running",
            "completed": False,
            "suite_plan_sha256": report_digest(plan),
            "stage": "preparation",
            "started_at_utc": started_at,
        }
        save_json(prior, status)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "event_caption_pilot.a4500_suite",
                    "check",
                    "--gpu-ids",
                    *map(str, args.gpu_ids),
                    "--minimum-free-gib",
                    "16",
                ],
                env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                check=True,
            )
            stage_started = perf_counter()
            execute_workers(worker_specs(root, plan, snapshot, prepare=True))
            prepared = prepared_records(root, plan, snapshot)
            preflight = selection_report(prepared, plan)
            save_json(root / "selection_preflight.json", preflight)
            load_prepared(root, plan, snapshot)
            save_json(
                root / "prepare_progress.json",
                {
                    "status": "prepared",
                    "prepared_videos": 20,
                    "total_videos": 20,
                },
            )
            status["preparation_stage_seconds"] = (
                perf_counter() - stage_started
            )
            print(preflight["comparisons"], flush=True)
            if args.stage == "prepare":
                status.update(status="prepared", stage="prepared")
            else:
                status["stage"] = "qa"
                save_json(prior, status)
                stage_started = perf_counter()

                def progress(done):
                    status["completed_jobs"] = done
                    save_json(prior, status)

                execute_workers(
                    worker_specs(root, plan, snapshot), on_complete=progress
                )
                aggregate(root)
                status.update(
                    status="complete",
                    completed=True,
                    completed_at_utc=now(),
                    qa_stage_seconds=perf_counter() - stage_started,
                )
        except BaseException as error:
            status.update(
                status="failed", error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            status["wall_seconds_this_invocation"] = perf_counter() - started
            save_json(prior, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
