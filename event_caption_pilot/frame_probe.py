"""Post-hoc OCR/change/QA probe on a human-selected pair of candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .backends import build_backend
from .cache import write_json
from .config import Config
from .data import array_digest, load_manifest
from .qa_diagnostics import report_digest
from .reproducibility import fix_seed


def run_probe(backend, video, question, indices, config):
    """Use independent prompts; never insert gold labels or earlier replies."""
    if (len(indices) != 2 or any(type(i) is not int for i in indices)
            or not 0 <= indices[0] < indices[1] < len(video.frames)):
        raise ValueError("Provide two increasing, valid candidate indices")
    single_prompt = (
        "Inspect the supplied image. Transcribe the visible handwritten "
        "letters on the paper exactly as displayed, from left to right. "
        "Do not complete a word or guess hidden letters. Use [unclear] for "
        "unreadable marks. State which letters are clearly readable."
    )
    tasks = [
        ("single_before", [indices[0]], single_prompt),
        ("single_after", [indices[1]], single_prompt),
        ("pair_change", indices,
         "Inspect the two images in timestamp order. First transcribe the "
         "visible handwritten letters in each image separately. Then "
         "describe only visible differences. Distinguish a directly visible "
         "writing action from a change inferred between images. If the "
         "images do not establish writing order, explicitly say so. Do not "
         "guess hidden or unclear letters."),
        ("pair_free_answer", indices,
         "Answer using only these two images in timestamp order. "
         "If the evidence is insufficient, say so. Briefly explain the "
         "visible evidence. Question: " + question.text),
    ]
    original_config = backend.config
    outputs = []
    try:
        for name, selected, prompt in tasks:
            backend.config = replace(
                config, caption_prompt=prompt,
                caption_prompt_version="frame-probe-v1-" + name,
                caption_max_frames=max(2, config.caption_max_frames),
                caption_max_new_tokens=256,
            )
            fix_seed(config.seed, config.deterministic, True,
                     config.seed_tensorflow, config.cublas_workspace_config)
            result = backend.caption(
                video.frames[selected].copy(),
                video.timestamps[selected].copy(),
            )
            outputs.append({
                "task": name, "indices": selected, "prompt": prompt,
                "generation": asdict(result),
                "output_token_limit_reached": result.output_tokens == 256,
            })
    finally:
        backend.config = original_config
    fix_seed(config.seed, config.deterministic, True,
             config.seed_tensorflow, config.cublas_workspace_config)
    qa = backend.answer(
        video.frames[indices].copy(), video.timestamps[indices].copy(),
        question.text, question.options,
    )
    return {
        "result_kind": "posthoc_human_selected_frame_probe_not_benchmark",
        "video_id": video.video_id, "question_id": question.question_id,
        "indices": indices, "timestamps": video.timestamps[indices].tolist(),
        "frame_shape": list(video.frames[indices].shape),
        "question": question.text, "options": list(question.options),
        "generations": outputs, "pair_original_qa_scoring": asdict(qa),
        "answer_index_evaluator_only": question.answer_index,
        "correct": qa.predicted_index == question.answer_index,
        "generation_max_new_tokens": 256,
        "note": "Independent prompts; no previous response, caption, or "
        "gold answer enters any model call. Human-selected frames and changed "
        "prompts make this a diagnostic, not a comparable benchmark arm.",
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
    if config.backend != "transformers" or config.mode != "real":
        raise ValueError("This probe requires a real transformers run")
    if config.encoder_revision == "main" or config.vlm_revision == "main":
        raise ValueError("Use pinned model revisions for the saved run")
    videos = load_manifest(config)
    video = next(v for v in videos if v.video_id == args.video_id)
    question = next(q for q in video.questions
                    if q.question_id == args.question_id)
    saved = next(v for v in report["videos"]
                 if v["video_id"] == video.video_id)
    if array_digest(video.frames, video.timestamps) != saved[
        "candidate_content_sha256"
    ]:
        raise ValueError("Candidate frames differ from the saved experiment")
    row = next(q for q in report["questions"]
               if q["video_id"] == video.video_id
               and q["question_id"] == question.question_id)
    if (row["question"] != question.text
            or row["options"] != list(question.options)
            or row["answer_index"] != question.answer_index):
        raise ValueError("Question differs from the saved experiment")
    if not 0 <= args.indices[0] < args.indices[1] < len(video.frames):
        raise ValueError("Indices must be increasing and in candidate range")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    backend = build_backend(config)
    result = run_probe(backend, video, question, args.indices, config)
    result["source_report_sha256"] = report_digest(report)
    result["config"] = config.to_dict()
    result["backend"] = backend.metadata()
    result["original_conditions"] = row["conditions"]
    write_json(args.output_dir / "probe.json", result)
    lines = ["# Two-frame diagnostic", "", result["note"], ""]
    for item in result["generations"]:
        lines.extend(["## " + item["task"], "",
                      item["generation"]["text"], ""])
    qa = result["pair_original_qa_scoring"]
    lines.extend(["## Original option scoring", "",
                  f"Prediction: {question.options[qa['predicted_index']]}",
                  f"Correct: {result['correct']}",
                  f"Scores: {qa['option_scores']}"])
    (args.output_dir / "probe.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved: {args.output_dir.resolve() / 'probe.md'}")


if __name__ == "__main__":
    main()
