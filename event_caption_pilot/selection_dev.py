"""Resume frame-selection experiments under an existing frozen QA protocol."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .ablation import select_frame_control
from .algorithms import normalize_rows
from .backends import build_backend
from .cache import write_json
from .config import Config
from .data import array_digest, decode_video, validate_video
from .frozen_dev import check_manifest
from .order_experiments import option_orders, score_call, seed_call
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .types import Question, Video

ARMS = ("uniform", "frame_top", "temporal_bin")


def summarize(rows):
    means = []
    for row in rows:
        accuracies = {
            arm: float(
                np.mean(
                    [
                        t["scoring_correct"]
                        for t in row["conditions"][arm]["trials"]
                    ]
                )
            )
            for arm in ARMS
        }
        means.append(
            {
                **{
                    k: row[k]
                    for k in ("video_id", "question_id", "duration_group")
                },
                **accuracies,
                "top_minus_uniform": accuracies["frame_top"]
                - accuracies["uniform"],
                "bin_minus_uniform": accuracies["temporal_bin"]
                - accuracies["uniform"],
            }
        )
    keys = (*ARMS, "top_minus_uniform", "bin_minus_uniform")
    return {
        "overall": clustered_means(means, keys),
        "by_duration": {
            v: clustered_means(
                [r for r in means if r["duration_group"] == v], keys
            )
            for v in sorted({r["duration_group"] for r in means})
        },
    }


def validate_baseline(config, manifest, source, locked):
    if locked["config"] != json.loads(json.dumps(config.to_dict())):
        raise ValueError("Baseline config differs")
    if locked["manifest_sha256"] != report_digest(manifest):
        raise ValueError("Baseline manifest differs")
    root = Path(__file__).parent
    for name in (
        "hf_backend.py",
        "order_experiments.py",
        "data.py",
        "algorithms.py",
        "answer_parser.py",
    ):
        if (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            != locked["code_sha256"][name]
        ):
            raise ValueError(f"Baseline code changed: {name}")
    if source["protocol_sha256"] != report_digest(locked):
        raise ValueError("Baseline protocol digest differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = json.loads(
        (args.baseline_dir / "results.json").read_text(encoding="utf-8")
    )
    baseline_protocol = json.loads(
        (args.baseline_dir / "protocol.json").read_text(encoding="utf-8")
    )
    settings = dict(baseline_protocol["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    manifest_path = Path(config.manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_baseline(config, manifest, source, baseline_protocol)
    dev = check_manifest(manifest)
    lookup = {
        (r["video_id"], r["question_id"]): r for r in source["questions"]
    }
    expected = {
        (v["video_id"], str(q["question_id"]))
        for v in dev
        for q in v["questions"]
    }
    if set(lookup) != expected:
        raise ValueError("Incomplete or extra baseline questions")
    lock = {
        "baseline_results_sha256": report_digest(source),
        "baseline_protocol_sha256": report_digest(baseline_protocol),
        "arms": ARMS,
        "primary_qa": "unchanged_full_option_log_likelihood",
        "code_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "selection_dev.py",
                "ablation.py",
                "paired_reanalysis.py",
            )
        },
    }
    digest = report_digest(lock)
    if args.resume:
        previous = json.loads(
            (args.output_dir / "protocol.json").read_text(encoding="utf-8")
        )
        if report_digest(previous) != digest:
            raise ValueError("Selection protocol changed")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "checkpoints").mkdir()
        write_json(args.output_dir / "protocol.json", lock)
    backend = None
    completed = []
    for item in dev:
        frames, times, duration = decode_video(
            manifest_path.parent / item["path"], config
        )
        questions = tuple(
            Question(
                str(q["question_id"]),
                q["text"],
                tuple(q["options"]),
                q["answer_index"],
            )
            for q in item["questions"]
        )
        video = Video(
            item["video_id"],
            "dev",
            frames,
            times,
            duration,
            questions,
            item["source_id"],
        )
        validate_video(video, config)
        content = array_digest(frames, times)
        features = None
        for question in questions:
            baseline = lookup[(video.video_id, question.question_id)]
            if (
                baseline["candidate_sha256"] != content
                or baseline["question"] != question.text
            ):
                raise ValueError("Baseline observations/question differ")
            orders = option_orders(len(question.options), config.seed)
            if [t["order"] for t in baseline["trials"]] != [
                list(o) for o in orders
            ]:
                raise ValueError("Baseline option orders differ")
            uniform = select_frame_control(
                "uniform", np.zeros(len(frames)), config.frame_budget
            ).tolist()
            if uniform != baseline["selected_indices"]:
                raise ValueError("Uniform indices differ")
            row = {
                "video_id": video.video_id,
                "question_id": question.question_id,
                "duration_group": item["duration"],
                "question_type": baseline["question_type"],
                "candidate_sha256": content,
                "conditions": {
                    "uniform": {
                        "selected_indices": uniform,
                        "trials": baseline["trials"],
                        "reused_baseline": True,
                    }
                },
            }
            for arm in ARMS[1:]:
                key = report_digest(
                    [video.video_id, question.question_id, arm]
                )
                checkpoint = args.output_dir / "checkpoints" / (key + ".json")
                if checkpoint.exists():
                    result = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if (
                        result["protocol_sha256"] != digest
                        or result["candidate_sha256"] != content
                    ):
                        raise ValueError("Checkpoint mismatch")
                else:
                    if backend is None:
                        # Set CUBLAS_WORKSPACE_CONFIG before the first CUDA
                        # allocation, not only immediately before answering.
                        seed_call(config)
                        backend = build_backend(config)
                        previous = json.loads(
                            (args.baseline_dir / "backend.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if backend.metadata() != previous:
                            raise ValueError(
                                "Runtime/model metadata differs from baseline"
                            )
                        write_json(
                            args.output_dir / "backend.json",
                            backend.metadata(),
                        )
                    if features is None:
                        features = normalize_rows(
                            backend.encode_frames(frames),
                            config.normalization_epsilon,
                        )
                    query = normalize_rows(
                        np.asarray(
                            backend.encode_visual_question(question.text)
                        )[None, :],
                        config.normalization_epsilon,
                    )[0]
                    scores = features @ query
                    indices = select_frame_control(
                        arm, scores, config.frame_budget
                    ).tolist()
                    trials = [
                        score_call(
                            backend, video, question, indices, o, config
                        )
                        for o in orders
                    ]
                    result = {
                        "protocol_sha256": digest,
                        "candidate_sha256": content,
                        "selected_indices": indices,
                        "selected_timestamps": times[indices].tolist(),
                        "frame_scores": scores.tolist(),
                        "trials": trials,
                    }
                    write_json(checkpoint, result)
                row["conditions"][arm] = result
                print(
                    f"Done {video.video_id}/{question.question_id}/{arm}",
                    flush=True,
                )
            completed.append(row)
            write_json(
                args.output_dir / "progress.json",
                {
                    "completed_questions": len(completed),
                    "total_questions": len(expected),
                },
            )
        del video, frames, times, features
    result = {
        "protocol_sha256": digest,
        "questions": completed,
        "summary": summarize(completed),
    }
    write_json(args.output_dir / "results.json", result)
    (args.output_dir / "summary.md").write_text(
        "# Fixed-QA selection experiment\n\n```json\n"
        + json.dumps(result["summary"], indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
