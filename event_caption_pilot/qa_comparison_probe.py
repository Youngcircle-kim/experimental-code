"""Same-prompt generation/scoring audit with sparse/dense observations."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

from .backends import build_backend
from .cache import write_json
from .config import Config
from .data import array_digest, load_manifest
from .qa_diagnostics import report_digest
from .reproducibility import fix_seed


def exact_option(text, options):
    """Conservative parsing: ambiguous/non-exact replies remain unparsed."""
    normalized = text.strip().casefold()
    matches = [i for i, option in enumerate(options)
               if option.strip().casefold() == normalized]
    return matches[0] if len(matches) == 1 else None


def run_comparison(backend, video, question, endpoints, config):
    if (len(endpoints) != 2 or any(type(i) is not int for i in endpoints)
            or not 0 <= endpoints[0] < endpoints[1] < len(video.frames)):
        raise ValueError("Expected two increasing candidate indices")
    if not 2 <= len(question.options) <= 4:
        raise ValueError(
            "Diagnostic supports 2–4 options for full permutations"
        )
    groups = {
        "sparse": list(endpoints),
        "dense": list(range(endpoints[0], endpoints[1] + 1)),
    }
    rows = []
    prior_audit = getattr(backend, "qa_audit_enabled", False)
    backend.qa_audit_enabled = True
    try:
        for group, indices in groups.items():
            for order in itertools.permutations(range(len(question.options))):
                options = tuple(question.options[i] for i in order)
                frames = video.frames[indices].copy()
                times = video.timestamps[indices].copy()
                fix_seed(
                    config.seed, config.deterministic, True,
                    config.seed_tensorflow, config.cublas_workspace_config,
                )
                generated = backend.generate_qa(
                    frames, times, question.text, options
                )
                fix_seed(
                    config.seed, config.deterministic, True,
                    config.seed_tensorflow, config.cublas_workspace_config,
                )
                scored = backend.answer(frames, times, question.text, options)
                audit = backend.last_qa_audit
                if (generated["prompt"] != audit["prompt"]
                        or generated["prompt_token_ids"]
                        != audit["prompt_token_ids"]):
                    raise ValueError("Generation and scoring prompts differ")
                parsed = exact_option(generated["text"], options)
                generated_original = None if parsed is None else order[parsed]
                scored_original = order[scored.predicted_index]
                rows.append({
                    "frame_group": group, "indices": indices,
                    "timestamps": times.tolist(),
                    "option_order_original_indices": list(order),
                    "presented_options": list(options),
                    "generation": generated, "scoring": asdict(scored),
                    "scoring_audit": audit, "same_prompt_verified": True,
                    "generated_original_index": generated_original,
                    "scored_original_index": scored_original,
                    "generation_parse_status": "exact_match"
                    if parsed is not None else "unparsed_or_ambiguous",
                    "generation_correct": None if parsed is None else
                    generated_original == question.answer_index,
                    "scoring_correct": (
                        scored_original == question.answer_index
                    ),
                    "methods_agree": None if parsed is None else
                    generated_original == scored_original,
                })
    finally:
        backend.qa_audit_enabled = prior_audit
    return {
        "result_kind": "posthoc_diagnostic_not_benchmark",
        "question": question.text, "options": list(question.options),
        "answer_index_evaluator_only": question.answer_index,
        "rows": rows,
        "limitations": [
            "Human-selected frames and one question cannot establish "
            "accuracy.",
            "Dense input has more frames; this is not an equal-budget test.",
            "Generation parsing accepts only exact option text ignoring case "
            "and outer whitespace; other replies require human inspection.",
            "Prompt text and token IDs are checked within each pair. Option "
            "permutations intentionally change the prompt between pairs.",
            "Identical prompts do not ensure identical numerical execution "
            "paths for generation and teacher-forced scoring.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--video-id", default="video_1726")
    parser.add_argument("--question-id", default="12")
    parser.add_argument("--indices", nargs=2, type=int, default=[6, 10])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    report = json.loads(args.results.read_text(encoding="utf-8-sig"))
    settings = dict(report["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    if (config.mode != "real" or config.backend != "transformers"
            or config.pilot != "both" or config.vlm_revision == "main"):
        raise ValueError("Use a real, pinned transformers QA run")
    video = next(v for v in load_manifest(config)
                 if v.video_id == args.video_id)
    question = next(q for q in video.questions
                    if q.question_id == args.question_id)
    saved = next(v for v in report["videos"]
                 if v["video_id"] == video.video_id)
    if array_digest(video.frames, video.timestamps) != saved[
        "candidate_content_sha256"
    ]:
        raise ValueError("Candidate frames differ from saved run")
    row = next(q for q in report["questions"]
               if q["video_id"] == video.video_id
               and q["question_id"] == question.question_id)
    if (row["question"] != question.text
            or row["options"] != list(question.options)
            or row["answer_index"] != question.answer_index):
        raise ValueError("Question differs from saved run")
    if not 0 <= args.indices[0] < args.indices[1] < len(video.frames):
        raise ValueError("Indices outside increasing candidate range")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backend = build_backend(config)
    result = run_comparison(backend, video, question, args.indices, config)
    result.update({
        "source_report_sha256": report_digest(report),
        "video_id": video.video_id, "question_id": question.question_id,
        "config": config.to_dict(), "backend": backend.metadata(),
    })
    write_json(args.output_dir / "comparison.json", result)
    lines = ["# Same-prompt QA and temporal-density diagnostic", "",
             "Sparse: endpoints only. Dense: every candidate between "
             "endpoints.",
             "", "| Frames | Option order | Generated text | Scored answer | "
             "Gen correct | Score correct | Agree |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in result["rows"]:
        text = item["generation"]["text"].replace("|", "\\|")
        text = text.replace("\n", " ").replace("\r", " ")
        predicted = question.options[item["scored_original_index"]]
        lines.append(
            f"| {item['frame_group']} | {item['presented_options']} | "
            f"{text} | {predicted} | {item['generation_correct']} | "
            f"{item['scoring_correct']} | {item['methods_agree']} |"
        )
    lines.extend(["", "## Limitations", ""] + result["limitations"])
    (args.output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved: {args.output_dir.resolve() / 'comparison.md'}")


if __name__ == "__main__":
    main()
