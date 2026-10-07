"""Length recheck, development order audit, and paired ablation evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ablation import COMPARISONS, CONDITIONS, HYPOTHESES
from .algorithms import sample_indices
from .answer_parser import parse_answer
from .backends import build_backend
from .cache import write_json
from .config import Config
from .data import array_digest, load_manifest
from .qa_diagnostics import report_digest
from .reproducibility import fix_seed


def option_orders(count, seed):
    """All six orders for three options; six fixed orders for four options."""
    if not 2 <= count <= 4:
        raise ValueError("Order protocol supports 2–4 options")
    orders = list(itertools.permutations(range(count)))
    if len(orders) > 6:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(np.arange(1, len(orders)), 5, replace=False)
        orders = [orders[0]] + [orders[int(i)] for i in sorted(chosen)]
    return orders


def seed_call(config):
    fix_seed(
        config.seed,
        config.deterministic,
        config.seed_torch or config.backend == "transformers",
        config.seed_tensorflow,
        config.cublas_workspace_config,
    )


def score_call(backend, video, question, indices, order, config):
    seed_call(config)
    options = tuple(question.options[i] for i in order)
    result = backend.answer(
        video.frames[indices].copy(),
        video.timestamps[indices].copy(),
        question.text,
        options,
    )
    scores = np.asarray(result.option_scores, dtype=float)
    if (
        scores.shape != (len(options),)
        or not np.isfinite(scores).all()
        or type(result.predicted_index) is not int
        or not 0 <= result.predicted_index < len(options)
    ):
        raise ValueError("Invalid QA output")
    original = order[result.predicted_index]
    return {
        "order": list(order),
        "presented_options": list(options),
        "scoring": asdict(result),
        "scored_original_index": original,
        "scoring_correct": original == question.answer_index,
    }


def paired_call(backend, video, question, indices, order, config, limit):
    options = tuple(question.options[i] for i in order)
    seed_call(config)
    generated = backend.generate_qa(
        video.frames[indices].copy(),
        video.timestamps[indices].copy(),
        question.text,
        options,
        max_new_tokens=limit,
    )
    prior = getattr(backend, "qa_audit_enabled", False)
    backend.qa_audit_enabled = True
    try:
        row = score_call(backend, video, question, indices, order, config)
        audit = backend.last_qa_audit
    finally:
        backend.qa_audit_enabled = prior
    if any(
        generated[key] != audit[key] for key in ("prompt", "prompt_token_ids")
    ):
        raise ValueError("Generation/scoring input prompts differ")
    parsing = parse_answer(
        generated["text"],
        options,
        generated["output_token_limit_reached"],
    )
    parsed = parsing["presented_index"]
    original = None if parsed is None else order[parsed]
    row.update(
        {
            "generation": generated,
            "generation_parsing": parsing,
            "answer_index_evaluator_only": question.answer_index,
            "scoring_audit": audit,
            "generated_original_index": original,
            "generation_correct": None
            if parsed is None
            else original == question.answer_index,
            "methods_agree": None
            if parsed is None
            else original == row["scored_original_index"],
        }
    )
    return row


def question_summary(trials):
    parsed = [r for r in trials if r["generated_original_index"] is not None]
    return {
        "n_orders": len(trials),
        "n_generation_parsed": len(parsed),
        "scoring_order_mean_accuracy": float(
            np.mean([r["scoring_correct"] for r in trials])
        ),
        "scoring_answer_changed": len(
            {r["scored_original_index"] for r in trials}
        )
        > 1,
        "generation_parse_rate": len(parsed) / len(trials),
        "generation_accuracy_on_parsed": float(
            np.mean([r["generation_correct"] for r in parsed])
        )
        if parsed
        else None,
        "generation_answer_changed_on_parsed": len(
            {r["generated_original_index"] for r in parsed}
        )
        > 1
        if len(parsed) >= 2
        else None,
        "method_disagreement_on_parsed": float(
            np.mean([not r["methods_agree"] for r in parsed])
        )
        if parsed
        else None,
        "generation_limit_rate": float(
            np.mean(
                [r["generation"]["output_token_limit_reached"] for r in trials]
            )
        ),
    }


def summarize_dev(rows):
    keys = list(
        question_summary(
            [
                {
                    "generated_original_index": None,
                    "scored_original_index": 0,
                    "scoring_correct": False,
                    "generation": {"output_token_limit_reached": False},
                }
            ]
        )
    )[2:]
    result = {
        "n_questions": len(rows),
        "n_videos": len({r["video_id"] for r in rows}),
    }
    for key in keys:
        values = [
            r["summary"][key] for r in rows if r["summary"][key] is not None
        ]
        result[key] = {
            "question_mean": float(np.mean(values)) if values else None,
            "eligible_questions": len(values),
        }
    return result


def aggregate_ablation(rows, config):
    """Average orders within question BEFORE video-cluster resampling."""
    if not rows:
        raise ValueError("No evaluation questions")
    seen = set()
    clusters = {}
    for row in rows:
        key = (row["video_id"], row["question_id"])
        if key in seen:
            raise ValueError("Duplicate question")
        seen.add(key)
        values = [row["condition_order_mean_accuracy"][c] for c in CONDITIONS]
        if not np.isfinite(values).all() or any(
            not 0 <= x <= 1 for x in values
        ):
            raise ValueError("Invalid question accuracy")
        clusters.setdefault(key[0], []).append(values)
    ids = sorted(clusters)
    sizes = np.array([len(clusters[i]) for i in ids])
    sums = np.array([np.sum(clusters[i], axis=0) for i in ids])
    accuracy = sums.sum(axis=0) / sizes.sum()
    bootstrap = None
    if len(ids) >= 2:
        draws = np.random.default_rng(config.seed).multinomial(
            len(ids),
            np.full(len(ids), 1 / len(ids)),
            size=config.bootstrap_samples,
        )
        bootstrap = (draws @ sums) / (draws @ sizes)[:, None]
    comparisons = {}
    alpha = (1 - config.confidence_level) / 2
    for treatment, control in COMPARISONS:
        a, b = CONDITIONS.index(treatment), CONDITIONS.index(control)
        comparisons[f"{treatment}_minus_{control}"] = {
            "mean_difference": float(accuracy[a] - accuracy[b]),
            "confidence_interval": None
            if bootstrap is None
            else np.quantile(
                bootstrap[:, a] - bootstrap[:, b], [alpha, 1 - alpha]
            ).tolist(),
        }
    return {
        "n_questions": len(rows),
        "n_videos": len(ids),
        "accuracies": dict(zip(CONDITIONS, accuracy.tolist())),
        "paired_comparisons": comparisons,
        "hypotheses": {
            name: {
                f"{a}_minus_{b}": comparisons[f"{a}_minus_{b}"]
                for a, b in pairs
            }
            for name, pairs in HYPOTHESES.items()
        },
        "aggregation": "equal order mean per question; question-weighted "
        "accuracy; paired original-video cluster bootstrap",
        "confidence_level": config.confidence_level,
        "bootstrap_samples": config.bootstrap_samples,
        "multiple_comparison_adjustment": "none",
    }


def validate_saved(report, videos):
    by_id = {v.video_id: v for v in videos}
    records = {v["video_id"]: v for v in report["videos"]}
    for row in report["questions"]:
        v = by_id[row["video_id"]]
        if (
            array_digest(v.frames, v.timestamps)
            != records[v.video_id]["candidate_content_sha256"]
        ):
            raise ValueError("Candidate pool differs from saved run")
        q = next(q for q in v.questions if q.question_id == row["question_id"])
        if (
            q.text != row["question"]
            or list(q.options) != row["options"]
            or q.answer_index != row["answer_index"]
        ):
            raise ValueError("Question differs from saved run")


def experiment_three(backend, report, videos, config):
    validate_saved(report, videos)
    by_id = {v.video_id: v for v in videos}
    rows = []
    for saved in report["questions"]:
        video = by_id[saved["video_id"]]
        if video.split != "eval":
            raise ValueError("Experiment 3 requires saved eval selections")
        question = next(
            q for q in video.questions if q.question_id == saved["question_id"]
        )
        orders = option_orders(len(question.options), config.seed)
        conditions = {}
        for name in CONDITIONS:
            indices = saved["conditions"][name]["selected_indices"]
            if (
                len(indices) != config.frame_budget
                or any(type(i) is not int for i in indices)
                or indices != sorted(set(indices))
                or not 0 <= indices[0] <= indices[-1] < len(video.frames)
            ):
                raise ValueError("Invalid saved exact-budget selection")
            conditions[name] = {
                "selected_indices": indices,
                "trials": [
                    score_call(
                        backend, video, question, indices, order, config
                    )
                    for order in orders
                ],
            }
        rows.append(
            {
                "video_id": video.video_id,
                "question_id": question.question_id,
                "conditions": conditions,
                "condition_order_mean_accuracy": {
                    name: float(
                        np.mean([t["scoring_correct"] for t in item["trials"]])
                    )
                    for name, item in conditions.items()
                },
            }
        )
    return {
        "questions": rows,
        "metrics": aggregate_ablation(rows, config),
        "qa_method": "fixed full-option conditional log likelihood",
        "source_report_sha256": report_digest(report),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("length", "dev", "ablation"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--previous-comparison", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest")
    parser.add_argument("--question-types", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    report = None
    if args.stage == "dev":
        if not args.config:
            parser.error("dev requires --config")
        settings = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if args.manifest:
            settings["manifest_path"] = args.manifest
    else:
        if not args.results:
            parser.error("length/ablation require --results")
        report = json.loads(args.results.read_text(encoding="utf-8-sig"))
        settings = dict(report["config"])
        if args.config or args.manifest:
            parser.error("Saved-run stages preserve the saved configuration")
    if "observation_strides" in settings:
        settings["observation_strides"] = tuple(
            settings["observation_strides"]
        )
    config = Config(**settings)
    config.validate()
    if (
        config.backend != "transformers"
        or config.mode != "real"
        or config.pilot != "both"
        or config.vlm_revision == "main"
    ):
        raise ValueError("Use a real transformers config with pinned VLM")
    videos = load_manifest(config)
    if report is not None:
        validate_saved(report, videos)
    previous = None
    selected = []
    if args.stage == "length":
        if not args.previous_comparison:
            parser.error("length requires --previous-comparison")
        previous = json.loads(
            args.previous_comparison.read_text(encoding="utf-8-sig")
        )
        if previous["source_report_sha256"] != report_digest(report):
            raise ValueError("Previous diagnostic belongs to a different run")
        selected = [
            r
            for r in previous["rows"]
            if r["frame_group"] == "sparse"
            and r["generation"]["output_token_limit_reached"]
        ]
        if not selected:
            raise ValueError("No sparse generations reached the token limit")
        if any(
            args.max_new_tokens <= r["generation"]["max_new_tokens"]
            for r in selected
        ):
            raise ValueError("New generation limit must exceed old limit")
    categories = {}
    if args.question_types:
        categories = json.loads(
            args.question_types.read_text(encoding="utf-8-sig")
        )
        if not isinstance(categories, dict) or any(
            not isinstance(x, str) for x in categories.values()
        ):
            raise ValueError(
                "Question types must map video_id/question_id to text"
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backend = build_backend(config)
    if args.stage == "ablation":
        result = experiment_three(backend, report, videos, config)
    elif args.stage == "dev":
        rows = []
        for video in videos:
            if video.split != "dev":
                continue
            indices = sample_indices(
                0, len(video.frames), config.frame_budget
            ).tolist()
            for question in video.questions:
                trials = [
                    paired_call(
                        backend,
                        video,
                        question,
                        indices,
                        order,
                        config,
                        args.max_new_tokens,
                    )
                    for order in option_orders(
                        len(question.options), config.seed
                    )
                ]
                rows.append(
                    {
                        "video_id": video.video_id,
                        "question_id": question.question_id,
                        "question_type": categories.get(
                            f"{video.video_id}/{question.question_id}",
                            "unlabeled",
                        ),
                        "question": question.text,
                        "selected_indices": indices,
                        "trials": trials,
                        "summary": question_summary(trials),
                    }
                )
        if not rows:
            raise ValueError("No development questions")
        result = {
            "questions": rows,
            "summary": summarize_dev(rows),
            "by_question_type": {
                kind: summarize_dev(
                    [r for r in rows if r["question_type"] == kind]
                )
                for kind in sorted({r["question_type"] for r in rows})
            },
            "scope": "dev only; fixed uniform candidates per question",
        }
    else:
        video = next(v for v in videos if v.video_id == previous["video_id"])
        question = next(
            q
            for q in video.questions
            if q.question_id == previous["question_id"]
        )
        trials = []
        for old in selected:
            new = paired_call(
                backend,
                video,
                question,
                old["indices"],
                old["option_order_original_indices"],
                config,
                args.max_new_tokens,
            )
            if any(
                new["generation"][k] != old["generation"][k]
                for k in ("prompt", "prompt_token_ids")
            ):
                raise ValueError("Recheck prompt differs from previous run")
            trials.append(
                {
                    "indices": old["indices"],
                    "previous_generation": old["generation"],
                    **new,
                }
            )
        result = {
            "trials": trials,
            "source_report_sha256": report_digest(report),
            "previous_comparison_sha256": report_digest(previous),
        }
    result.update(
        {
            "stage": args.stage,
            "config": config.to_dict(),
            "backend": backend.metadata(),
            "max_new_tokens": args.max_new_tokens,
            "note": "Order repeats are not independent questions. "
            "Unparsed generation is unavailable, not automatically wrong. "
            "No evaluation-driven tuning or automatic protocol selection.",
        }
    )
    write_json(args.output_dir / "results.json", result)
    summary = result.get(
        "metrics", result.get("summary", result.get("trials"))
    )
    (args.output_dir / "summary.md").write_text(
        f"# {args.stage} experiment\n\n{result['note']}\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
