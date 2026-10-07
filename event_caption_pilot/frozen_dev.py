"""Stream a locked dev protocol with per-question resumable checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

from .algorithms import sample_indices
from .answer_parser import PARSER_VERSION
from .backends import build_backend
from .cache import write_json
from .config import Config
from .data import array_digest, decode_video, validate_video
from .order_experiments import (
    option_orders,
    paired_call,
    question_summary,
    score_call,
    summarize_dev,
)
from .qa_diagnostics import report_digest
from .types import Question, Video


def protocol(config, manifest, with_generation):
    directory = Path(__file__).parent
    return {
        "version": 1,
        "config": config.to_dict(),
        "manifest_sha256": report_digest(manifest),
        "split": "dev",
        "primary_qa": "full_option_log_likelihood",
        "selection": "fixed_uniform_16"
        if config.frame_budget == 16
        else f"fixed_uniform_{config.frame_budget}",
        "with_generation": with_generation,
        "generation_max_new_tokens": 256,
        "parser_version": PARSER_VERSION,
        "subtitles": False,
        "audio": False,
        "orders": {str(n): option_orders(n, config.seed) for n in (2, 3, 4)},
        "code_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "hf_backend.py",
                "order_experiments.py",
                "answer_parser.py",
                "data.py",
                "algorithms.py",
                "frozen_dev.py",
            )
        },
        "interpretation": "Custom dev audit, not official leaderboard. "
        "Order repeats remain within questions. Holdout is not inferred on.",
    }


def check_manifest(manifest):
    ids, sources = set(), set()
    for v in manifest["videos"]:
        if v["video_id"] in ids or v["source_id"] in sources:
            raise ValueError("Duplicate video/source across manifest")
        ids.add(v["video_id"])
        sources.add(v["source_id"])
        if v["split"] not in ("dev", "eval"):
            raise ValueError("Unknown split")
    dev = [v for v in manifest["videos"] if v["split"] == "dev"]
    if not dev:
        raise ValueError("No development videos")
    return dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8-sig"))
    settings["observation_strides"] = tuple(
        settings.get("observation_strides", [1, 2, 4])
    )
    config = Config(**settings)
    config.validate()
    if (
        config.mode != "real"
        or config.backend != "transformers"
        or config.pilot != "both"
        or config.vlm_revision == "main"
    ):
        raise ValueError("Use a pinned real transformers config")
    manifest_path = Path(config.manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    dev = check_manifest(manifest)
    missing = [
        str(manifest_path.parent / v["path"])
        for v in dev
        if not (manifest_path.parent / v["path"]).is_file()
    ]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} dev videos; first: {missing[0]}"
        )
    locked = protocol(config, manifest, args.with_generation)
    digest = report_digest(locked)
    if args.resume:
        old = json.loads((args.output_dir / "protocol.json").read_text())
        if report_digest(old) != digest:
            raise ValueError(
                "Config, data annotations, or code changed; new run required"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "protocol.json", locked)
        (args.output_dir / "questions").mkdir()
    backend = None
    rows = []
    content_seen = set()
    for item in dev:
        path = manifest_path.parent / item["path"]
        if item.get("timestamp_mode") != "constant_fps":
            raise ValueError("This runner requires constant-FPS MP4 manifests")
        frames, times, duration = decode_video(path, config)
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
        if content in content_seen:
            raise ValueError("Duplicate decoded development video content")
        content_seen.add(content)
        indices = sample_indices(0, len(frames), config.frame_budget).tolist()
        for q, annotation in zip(questions, item["questions"]):
            key = hashlib.sha256(
                json.dumps([video.video_id, q.question_id]).encode()
            ).hexdigest()
            checkpoint = args.output_dir / "questions" / (key + ".json")
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (
                    row["protocol_sha256"] != digest
                    or row["candidate_sha256"] != content
                ):
                    raise ValueError("Checkpoint input/protocol mismatch")
            else:
                if backend is None:
                    backend = build_backend(config)
                    write_json(
                        args.output_dir / "backend.json", backend.metadata()
                    )
                trials = []
                for order in option_orders(len(q.options), config.seed):
                    if args.with_generation:
                        trial = paired_call(
                            backend, video, q, indices, order, config, 256
                        )
                    else:
                        trial = score_call(
                            backend, video, q, indices, order, config
                        )
                    trials.append(trial)
                row = {
                    "video_id": video.video_id,
                    "question_id": q.question_id,
                    "question": q.text,
                    "question_type": annotation["task_type"],
                    "duration_group": item["duration"],
                    "domain": item["domain"],
                    "candidate_sha256": content,
                    "protocol_sha256": digest,
                    "selected_indices": indices,
                    "trials": trials,
                    "scoring_order_mean_accuracy": sum(
                        t["scoring_correct"] for t in trials
                    )
                    / len(trials),
                    "scoring_answer_changed": len(
                        {t["scored_original_index"] for t in trials}
                    )
                    > 1,
                }
                if args.with_generation:
                    row["summary"] = question_summary(trials)
                write_json(checkpoint, row)
            rows.append(row)
            print(
                f"Completed {len(rows)}: {video.video_id}/{q.question_id}",
                flush=True,
            )
        del video, frames, times

    def summarize(subset):
        return {
            "n_questions": len(subset),
            "n_videos": len({r["video_id"] for r in subset}),
            "scoring_order_mean_accuracy": sum(
                r["scoring_order_mean_accuracy"] for r in subset
            )
            / len(subset),
            "answer_change_rate": sum(
                r["scoring_answer_changed"] for r in subset
            )
            / len(subset),
        }

    summary = summarize(rows)
    groups = {
        field: {
            value: summarize([r for r in rows if r[field] == value])
            for value in sorted({r[field] for r in rows})
        }
        for field in ("question_type", "duration_group", "domain")
    }
    result = {
        "protocol_sha256": digest,
        "summary": summary,
        "groups": groups,
        "questions": rows,
    }
    if args.with_generation:
        result["generation_summary"] = summarize_dev(rows)
    write_json(args.output_dir / "results.json", result)
    (args.output_dir / "summary.md").write_text(
        "# Frozen Video-MME dev audit\n\n```json\n"
        + json.dumps({"summary": summary, "groups": groups}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
