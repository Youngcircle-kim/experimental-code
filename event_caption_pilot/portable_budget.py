"""Independent, resumable 16/32-frame video QA experiment for one GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .ablation import select_frame_control
from .algorithms import normalize_rows
from .backends import build_backend
from .compare_budgets import validate_backend
from .config import Config
from .data import array_digest, decode_video, validate_video
from .event_factorial import make_partitions, select_plans
from .event_factorial_dev import validate_indices
from .event_refinement_dev import question_from
from .frame_replacement_probe import strict_trials
from .order_experiments import option_orders, seed_call
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .resolution_dev import (
    decode_selected,
    score_with_memory,
    validate_processor,
)
from .types import Video

CONDITIONS = ("uniform", "frame_top", "temporal_bin", "C", "D")
COMPARISONS = {
    "D_minus_uniform": ("D", "uniform"),
    "D_minus_C": ("D", "C"),
    "D_minus_frame_top": ("D", "frame_top"),
    "D_minus_temporal_bin": ("D", "temporal_bin"),
}
QA_SIDE = 320


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Also rejects JSON's accepted NaN/Infinity and overflowing exponents.
    json.dumps(value, allow_nan=False)
    return value


def save_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_identity(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"Input changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def runtime_metadata():
    versions = {}
    for package in (
        "numpy",
        "torch",
        "torchvision",
        "transformers",
        "Pillow",
        "opencv-python",
        "huggingface-hub",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


@contextmanager
def output_lock(directory):
    """Kernel-released advisory lock; never delete another process's lock."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"Another process is writing this output: {directory}"
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def checkpoint(path, identity, compute=None):
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity or saved[
            "payload_sha256"
        ] != report_digest(saved["payload"]):
            raise ValueError(f"Checkpoint identity/content mismatch: {path}")
        return saved["payload"]
    if compute is None:
        raise ValueError(f"Missing checkpoint: {path}")
    # JSON converts integer object keys to strings (e.g. decoded frame-size
    # metadata). Hash its persisted representation, not the pre-save object.
    value = json.loads(json.dumps(compute(), allow_nan=False))
    save_json(
        path,
        {
            "identity": identity,
            "payload": value,
            "payload_sha256": report_digest(value),
        },
    )
    return value


def load_inputs(args):
    settings = read_json(args.config)
    if "observation_strides" in settings:
        settings["observation_strides"] = tuple(
            settings["observation_strides"]
        )
    manifest_path = Path(args.manifest or settings["manifest_path"]).resolve()
    config = replace(
        Config(**settings),
        frame_budget=args.frame_budget,
        manifest_path=str(manifest_path),
    )
    config.validate()
    if (
        config.mode != "real"
        or config.backend != "transformers"
        or config.pilot != "both"
        or config.candidate_fps != 2.0
        or config.frame_height != 224
        or config.frame_width != 224
        or config.vlm_model != "Qwen/Qwen3.5-4B"
        or config.encoder_model != "openai/clip-vit-base-patch32"
        or config.vlm_min_pixels != QA_SIDE**2
        or config.vlm_max_pixels != QA_SIDE**2
        or config.device != "cuda:0"
        or config.model_dtype != "bfloat16"
        or config.attention_implementation != "sdpa"
        or not config.qa_length_normalize
        or not config.deterministic
        or config.bootstrap_samples != 5000
        or config.confidence_level != 0.95
        or any(
            not re.fullmatch(r"[0-9a-f]{40}", revision)
            for revision in (
                config.encoder_revision,
                config.vlm_revision,
            )
        )
    ):
        raise ValueError(
            "Require pinned (40-hex revisions) CLIP + Qwen3.5-4B, real "
            "cuda:0 BF16 SDPA; 2 FPS 224 candidates, 320 QA, deterministic "
            "length-normalized scoring and 5000 bootstrap samples. "
            "Use prepare_models with configs/real.a4500.json first."
        )
    if args.frame_budget not in (16, 32):
        raise ValueError("frame-budget must be 16 or 32")
    if args.split == "eval" and (
        args.max_videos is not None or args.max_questions is not None
    ):
        raise ValueError("Evaluation forbids subset limits; use dev smoke")
    for limit in (args.max_videos, args.max_questions):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Subset limits must be positive integers")
    manifest = read_json(manifest_path)
    seen_ids, seen_sources, seen_paths = set(), set(), set()
    for item in manifest["videos"]:
        vid, source = item["video_id"], item["source_id"]
        path = str((manifest_path.parent / item["path"]).resolve())
        if (
            not isinstance(vid, str)
            or not vid
            or not isinstance(source, str)
            or not source
            or vid in seen_ids
            or source in seen_sources
            or path in seen_paths
        ):
            raise ValueError("Require globally unique video/source IDs/paths")
        seen_ids.add(vid)
        seen_sources.add(source)
        seen_paths.add(path)
        if (
            item["split"] not in {"dev", "eval"}
            or item["duration"] not in {"short", "medium", "long"}
            or item.get("timestamp_mode") != "constant_fps"
            or Path(path).suffix.lower() == ".npz"
        ):
            raise ValueError("Require duration groups and original CFR videos")
        qids = set()
        if not item["questions"]:
            raise ValueError("Every video requires questions")
        for annotation in item["questions"]:
            q = question_from(annotation)
            if (
                not q.question_id
                or q.question_id in qids
                or not isinstance(q.text, str)
                or not q.text.strip()
                or not 2 <= len(q.options) <= 4
                or any(
                    not isinstance(o, str) or not o.strip() for o in q.options
                )
                or type(q.answer_index) is not int
                or not 0 <= q.answer_index < len(q.options)
            ):
                raise ValueError("Invalid question ID/text/options/answer")
            qids.add(q.question_id)
    chosen = [
        {**v, "questions": v["questions"][: args.max_questions]}
        for v in manifest["videos"]
        if v["split"] == args.split
        and (args.duration == "all" or v["duration"] == args.duration)
    ][: args.max_videos]
    if not chosen:
        raise ValueError("No videos match the requested split/duration")
    inputs = {
        v["video_id"]: file_identity(manifest_path.parent / v["path"])
        for v in chosen
    }
    if len({i["sha256"] for i in inputs.values()}) != len(inputs):
        raise ValueError("Duplicate source file content in selected cohort")
    protocol = {
        "version": 1,
        "experiment": "portable_frame_budget_v1",
        "config": config.to_dict(),
        "config_source_sha256": report_digest(settings),
        "manifest_sha256": report_digest(manifest),
        "manifest_path": str(manifest_path),
        "qa_side": QA_SIDE,
        "split": args.split,
        "duration": args.duration,
        "max_videos": args.max_videos,
        "max_questions": args.max_questions,
        "conditions": list(CONDITIONS),
        "primary_comparison": "D_minus_uniform",
        "secondary_comparisons": list(COMPARISONS)[1:],
        "cohort": {
            v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
            for v in chosen
        },
        "inputs": inputs,
        "orders_by_option_count": {
            str(n): [list(o) for o in option_orders(n, config.seed)]
            for n in (2, 3, 4)
        },
        "code_sha256": {
            p.name: hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    # Normalize tuples now so resumed JSON compares without type differences.
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    return config, chosen, protocol


def make_selection(features, query, timestamps, partitions, config):
    """The only selector inputs: pixels' features, query text, timestamps."""
    features = normalize_rows(features, config.normalization_epsilon)
    query = normalize_rows(
        np.asarray(query)[None, :], config.normalization_epsilon
    )[0]
    event_plans = select_plans(features, query, timestamps, partitions, config)
    plans = {}
    for name in CONDITIONS:
        if name in {"C", "D"}:
            plans[name] = event_plans[name]
        else:
            indices = select_frame_control(
                name, features @ query, config.frame_budget
            )
            plans[name] = {
                "selected_indices": indices.tolist(),
                "selected_timestamps": timestamps[indices].tolist(),
                "selection": name,
            }
    return plans


def summarize_rows(rows, seed=42):
    if not rows:
        raise ValueError("No completed questions")
    means = []
    for row in rows:
        if set(row["conditions"]) != set(CONDITIONS):
            raise ValueError("Incomplete paired conditions")
        reference = [
            t["order"] for t in row["conditions"]["uniform"]["trials"]
        ]
        if not reference or len({tuple(o) for o in reference}) != len(
            reference
        ):
            raise ValueError("Empty or repeated option orders")
        values = {}
        for name in CONDITIONS:
            trials = row["conditions"][name]["trials"]
            if [t["order"] for t in trials] != reference or any(
                type(t["scoring_correct"]) is not bool for t in trials
            ):
                raise ValueError("Paired order/correctness mismatch")
            values[name] = float(
                np.mean([t["scoring_correct"] for t in trials])
            )
        values.update(
            {k: values[a] - values[b] for k, (a, b) in COMPARISONS.items()}
        )
        means.append(
            {
                **{
                    k: row[k]
                    for k in ("video_id", "question_id", "duration_group")
                },
                **values,
            }
        )
    keys = [*CONDITIONS, *COMPARISONS]

    def resources(trials):
        timings = [
            t["qa_wall_seconds"] for t in trials if "qa_wall_seconds" in t
        ]
        result = {
            "n_trials": len(trials),
            "median_qa_seconds": float(np.median(timings))
            if timings
            else None,
        }
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            peaks = [
                t["cuda_memory"][key]
                for t in trials
                if t.get("cuda_memory", {}).get(key) is not None
            ]
            result["max_cuda_" + key] = max(peaks) if peaks else None
        return result

    arm_resources = {
        name: resources(
            [t for row in rows for t in row["conditions"][name]["trials"]]
        )
        for name in CONDITIONS
    }
    all_resources = resources(
        [
            t
            for row in rows
            for name in CONDITIONS
            for t in row["conditions"][name]["trials"]
        ]
    )
    return {
        "primary_comparison": "D_minus_uniform",
        "overall": clustered_means(means, keys, seed=seed, samples=5000),
        "by_duration": {
            g: clustered_means(
                [m for m in means if m["duration_group"] == g],
                keys,
                seed=seed,
                samples=5000,
            )
            for g in sorted({m["duration_group"] for m in means})
        },
        "paired_question_changes": {
            k: {
                "improved": sum(m[k] > 1e-12 for m in means),
                "degraded": sum(m[k] < -1e-12 for m in means),
                "unchanged": sum(abs(m[k]) <= 1e-12 for m in means),
            }
            for k in COMPARISONS
        },
        "question_metrics": means,
        "qa_resources": {
            **all_resources,
            "by_condition": arm_resources,
            "scope": "QA preprocessing/scoring only, not end-to-end. "
            "CUDA peaks measure the PyTorch allocator including resident "
            "CLIP and QA model weights; exclude decoder/selection CPU RAM "
            "and other processes. Timing reflects cached original trials "
            "when resuming; no new inference is implied.",
        },
        "interpretation": (
            "Order means within questions; video-cluster percentile 95% CIs. "
            "Secondary comparisons are exploratory without multiplicity "
            "correction. D-C changes segment boundaries, lengths and pooled "
            "features. No event captions, audio, subtitles or evidence labels "
            "enter selection. A single-video smoke cannot estimate "
            "uncertainty."
        ),
    }


def validate_completed_row(row, item, question, config, digest, output):
    expected = {
        "video_id": item["video_id"],
        "source_id": item["source_id"],
        "question_id": question.question_id,
        "duration_group": item["duration"],
        "question": question.text,
        "options": list(question.options),
        "answer_index_evaluator_only": question.answer_index,
    }
    if any(row[k] != value for k, value in expected.items()) or set(
        row["conditions"]
    ) != set(CONDITIONS):
        raise ValueError("Completed question identity/conditions mismatch")
    plan = checkpoint(
        output / "plans" / (report_digest(item["video_id"]) + ".json"),
        {
            "protocol_sha256": digest,
            "video_id": item["video_id"],
            "candidate_sha256": row["candidate_sha256"],
        },
    )
    if row["candidate_count"] != plan["candidate_count"]:
        raise ValueError("Completed candidate count mismatch")
    for name in CONDITIONS:
        condition = row["conditions"][name]
        selection = plan["questions"][question.question_id][name]
        if any(condition[k] != value for k, value in selection.items()):
            raise ValueError("Completed selection/plan mismatch")
        indices = condition["selected_indices"]
        validate_indices(indices, row["candidate_count"], config.frame_budget)
        stamps = np.asarray(condition["selected_timestamps"], dtype=float)
        if (
            stamps.shape != (config.frame_budget,)
            or not np.isfinite(stamps).all()
            or np.any(np.diff(stamps) <= 0)
            or not 0 <= stamps[0] < stamps[-1] < plan["duration_seconds"]
        ):
            raise ValueError("Completed selection timestamp mismatch")
        strict_trials(condition["trials"], question, config)
        for trial in condition["trials"]:
            validate_processor(trial, QA_SIDE, config.frame_budget)
            trial_id = {
                "protocol_sha256": digest,
                "video_id": item["video_id"],
                "question_id": question.question_id,
                "condition": name,
                "indices": indices,
                "order": trial["order"],
                "qa_pixels_sha256": condition["qa_pixels_sha256"],
            }
            saved = checkpoint(
                output / "trials" / (report_digest(trial_id) + ".json"),
                trial_id,
            )
            if saved != trial:
                raise ValueError(
                    "Completed question/trial checkpoint mismatch"
                )


def run(args):
    config, chosen, protocol = load_inputs(args)
    total = sum(len(v["questions"]) for v in chosen)
    digest = report_digest(protocol)
    if args.check_inputs:
        status = {
            "status": "inputs_validated_no_inference",
            "n_videos": len(chosen),
            "n_questions": total,
            "frame_budget": config.frame_budget,
            "maximum_qa_trials": sum(
                len(option_orders(len(q["options"]), config.seed))
                * len(CONDITIONS)
                for v in chosen
                for q in v["questions"]
            ),
            "protocol_sha256": digest,
        }
        print(json.dumps(status))
        return status
    output = args.output_dir.resolve()
    with output_lock(output):
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if read_json(protocol_path) != protocol:
                raise ValueError(
                    "Protocol/input/code changed; use a new output directory"
                )
        else:
            if args.summarize_only or any(
                p.name != ".run.lock" for p in output.iterdir()
            ):
                raise ValueError("Output has no compatible protocol")
            save_json(protocol_path, protocol)
        # seed_call sets this before CUDA initialization; fingerprint its
        # effective value so the first run and its resume agree.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
        runtime = runtime_metadata()
        runtime_path = output / "runtime.json"
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError(
                "Runtime environment changed; use a new output directory"
            )
        save_json(runtime_path, runtime)
        metadata_path = output / "backend.json"
        backend_identity = {"protocol_sha256": digest}
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            validate_backend(metadata, config)
            recorded = checkpoint(
                output / "backend_identity.json", backend_identity
            )
            if recorded != metadata:
                raise ValueError("Backend metadata integrity mismatch")
        model = None

        def get_model():
            nonlocal model
            if model is None:
                seed_call(config)
                model = build_backend(config)
                metadata = model.metadata()
                validate_backend(metadata, config)
                if (
                    metadata_path.exists()
                    and read_json(metadata_path) != metadata
                ):
                    raise ValueError("Backend metadata changed; cannot resume")
                recorded = checkpoint(
                    output / "backend_identity.json",
                    backend_identity,
                    lambda: metadata,
                )
                if recorded != metadata:
                    raise ValueError("Backend metadata changed; cannot resume")
                save_json(metadata_path, metadata)
            return model

        def qpath(vid, qid):
            return output / "questions" / (report_digest([vid, qid]) + ".json")

        rows, seen_candidates = [], set()
        for item in chosen:
            vid = item["video_id"]
            questions = tuple(question_from(a) for a in item["questions"])
            all_complete = all(
                qpath(vid, q.question_id).exists() for q in questions
            )
            if not all_complete:
                if args.summarize_only:
                    raise ValueError(
                        "Cannot summarize: question checkpoints incomplete"
                    )
                source_path = Path(protocol["inputs"][vid]["path"])
                started = perf_counter()
                frames, times, duration = decode_video(source_path, config)
                video = Video(
                    vid,
                    args.split,
                    frames,
                    times,
                    duration,
                    questions,
                    item["source_id"],
                )
                validate_video(video, config)
                candidate_hash = array_digest(frames, times)
                identity = {
                    "protocol_sha256": digest,
                    "video_id": vid,
                    "candidate_sha256": candidate_hash,
                }

                def prepare():
                    backend = get_model()
                    features = normalize_rows(
                        backend.encode_frames(frames),
                        config.normalization_epsilon,
                    )
                    partitions = make_partitions(features, config)
                    return {
                        "candidate_sha256": candidate_hash,
                        "candidate_count": len(times),
                        "duration_seconds": duration,
                        "questions": {
                            q.question_id: make_selection(
                                features,
                                backend.encode_visual_question(q.text),
                                times,
                                partitions,
                                config,
                            )
                            for q in questions
                        },
                        "prepare_wall_seconds": perf_counter() - started,
                    }

                plan = checkpoint(
                    output / "plans" / (report_digest(vid) + ".json"),
                    identity,
                    prepare,
                )
                # Free 224 candidate RGB before retaining 320 QA frames.
                del video, frames
                for q in questions:
                    if qpath(vid, q.question_id).exists():
                        continue
                    selections = plan["questions"][q.question_id]
                    for selection in selections.values():
                        validate_indices(
                            selection["selected_indices"],
                            len(times),
                            config.frame_budget,
                        )
                        if (
                            selection["selected_timestamps"]
                            != times[selection["selected_indices"]].tolist()
                        ):
                            raise ValueError("Selection timestamps mismatch")
                    union = sorted(
                        {
                            i
                            for s in selections.values()
                            for i in s["selected_indices"]
                        }
                    )
                    low, pixels, replay_times, replay_duration, decode_info = (
                        decode_selected(
                            source_path,
                            config,
                            len(times),
                            union,
                            QA_SIDE,
                            candidate_hash,
                        )
                    )
                    del low
                    conditions = {}
                    execution = list(CONDITIONS)
                    if config.randomize_condition_order:
                        np.random.default_rng(
                            config.seed
                            + int(report_digest([vid, q.question_id])[:8], 16)
                        ).shuffle(execution)
                    for name in execution:
                        selected = selections[name]
                        indices = selected["selected_indices"]
                        qa_frames = np.stack([pixels[i] for i in indices])
                        qa_times = replay_times[indices]
                        qa_video = Video(
                            vid,
                            args.split,
                            qa_frames,
                            qa_times,
                            replay_duration,
                            (),
                            item["source_id"],
                        )
                        pixel_hash = array_digest(qa_frames, qa_times)
                        trials = []
                        for order in option_orders(
                            len(q.options), config.seed
                        ):
                            trial_id = {
                                "protocol_sha256": digest,
                                "video_id": vid,
                                "question_id": q.question_id,
                                "condition": name,
                                "indices": indices,
                                "order": list(order),
                                "qa_pixels_sha256": pixel_hash,
                            }
                            trial = checkpoint(
                                output
                                / "trials"
                                / (report_digest(trial_id) + ".json"),
                                trial_id,
                                lambda: score_with_memory(
                                    get_model(),
                                    qa_video,
                                    q,
                                    order,
                                    config,
                                    QA_SIDE,
                                ),
                            )
                            validate_processor(
                                trial, QA_SIDE, config.frame_budget
                            )
                            trials.append(trial)
                        strict_trials(trials, q, config)
                        conditions[name] = {
                            **selected,
                            "qa_pixels_sha256": pixel_hash,
                            "trials": trials,
                        }
                        del qa_video, qa_frames
                    del pixels
                    row = {
                        "video_id": vid,
                        "source_id": item["source_id"],
                        "question_id": q.question_id,
                        "duration_group": item["duration"],
                        "candidate_sha256": candidate_hash,
                        "candidate_count": len(times),
                        "question": q.text,
                        "options": list(q.options),
                        "answer_index_evaluator_only": q.answer_index,
                        "execution_order": execution,
                        "decode_info": decode_info,
                        "conditions": {n: conditions[n] for n in CONDITIONS},
                    }
                    checkpoint(
                        qpath(vid, q.question_id),
                        {
                            "protocol_sha256": digest,
                            "video_id": vid,
                            "question_id": q.question_id,
                        },
                        lambda: row,
                    )
                    print(
                        f"Saved {vid}/{q.question_id} "
                        f"({config.frame_budget} frames)",
                        flush=True,
                    )
            video_hashes = set()
            for q in questions:
                row = checkpoint(
                    qpath(vid, q.question_id),
                    {
                        "protocol_sha256": digest,
                        "video_id": vid,
                        "question_id": q.question_id,
                    },
                )
                validate_completed_row(row, item, q, config, digest, output)
                video_hashes.add(row["candidate_sha256"])
                rows.append(row)
            if len(video_hashes) != 1 or video_hashes & seen_candidates:
                raise ValueError(
                    "Duplicate/inconsistent decoded candidate content"
                )
            seen_candidates.update(video_hashes)
            save_json(
                output / "progress.json",
                {
                    "status": "running",
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "protocol_sha256": digest,
                },
            )
        summary = summarize_rows(rows, config.seed)
        results = {
            "protocol_sha256": digest,
            "completed": True,
            "config": config.to_dict(),
            "questions": rows,
            "summary": summary,
        }
        save_json(output / "results.json", results)
        metrics = summary["question_metrics"]
        with (output / "question_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        save_json(
            output / "progress.json",
            {
                "status": "complete",
                "completed_questions": total,
                "total_questions": total,
                "protocol_sha256": digest,
            },
        )
        save_json(
            output / "handoff.json",
            {
                "status": "complete",
                "protocol_sha256": digest,
                "split": args.split,
                "duration": args.duration,
                "frame_budget": config.frame_budget,
                "n_videos": len(chosen),
                "n_questions": total,
                "summary": summary,
                "outputs": [
                    "results.json",
                    "protocol.json",
                    "runtime.json",
                    "backend.json",
                    "question_metrics.csv",
                ],
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(output),
                    "overall": summary["overall"],
                },
                indent=2,
            )
        )
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frame-budget", type=int, choices=(16, 32), required=True
    )
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument(
        "--duration",
        choices=("all", "short", "medium", "long"),
        default="long",
    )
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-questions", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-inputs", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
