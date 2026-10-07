"""Post-hoc single-frame replacement with an unchanged QA protocol."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .backends import build_backend
from .bottleneck_diagnostics import read, validate_plan, write_csv
from .cache import write_json
from .config import Config
from .data import array_digest, validate_video
from .event_factorial_dev import (
    cached_trial,
    measured_score,
    validate_indices,
    validate_trials,
)
from .midpoint_review import load_candidates
from .order_experiments import seed_call
from .qa_diagnostics import report_digest
from .types import Question, Video

FROZEN_CODE = (
    "hf_backend.py",
    "backends.py",
    "order_experiments.py",
    "event_factorial_dev.py",
    "data.py",
    "config.py",
    "reproducibility.py",
    "algorithms.py",
)


def variants(plan, remove_index, replacements):
    """Replace exactly one single-midpoint frame within its existing event."""
    original = plan["selected_indices"]
    if type(remove_index) is not int or remove_index not in original:
        raise ValueError("Removed frame must belong to the saved D input")
    if not replacements or len(set(replacements)) != len(replacements):
        raise ValueError("Supply unique replacement indices")
    event_id = next(
        i for i, (a, b) in enumerate(plan["events"]) if a <= remove_index < b
    )
    start, stop = plan["events"][event_id]
    if (
        plan["allocation"][event_id] != 1
        or remove_index != start + (stop - start - 1) // 2
    ):
        raise ValueError(
            "Removed frame must be the sole midpoint of its event"
        )
    result = {"rerun_D": list(original)}
    for index in replacements:
        if type(index) is not int or not start <= index < stop:
            raise ValueError(
                "Replacement must be a candidate in the same event"
            )
        if index in original:
            raise ValueError("Replacement must not already be selected")
        selected = sorted(index if i == remove_index else i for i in original)
        validate_indices(selected, plan["events"][-1][1], len(original))
        if set(original) - set(selected) != {remove_index} or set(
            selected
        ) - set(original) != {index}:
            raise ValueError("Exactly one frame must change")
        result[f"replace_{remove_index}_with_{index}"] = selected
    return event_id, result


def strict_trials(trials, question, config):
    validate_trials(trials, question, config)
    if any(
        t["scoring"]["predicted_index"]
        != int(np.argmax(t["scoring"]["option_scores"]))
        for t in trials
    ):
        raise ValueError("Prediction must maximize the recorded option scores")


def load_inputs(
    source,
    video_id,
    question_id,
    remove_index,
    replacements,
    baseline_dir=None,
):
    source = Path(source).resolve()
    protocol = read(source / "protocol.json")
    root = Path(__file__).parent
    for name in FROZEN_CODE:
        if (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            != protocol["code_sha256"][name]
        ):
            raise ValueError(f"Original QA/decode code changed: {name}")
    settings = dict(protocol["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    if (
        config.mode != "real"
        or config.backend != "transformers"
        or config.pilot != "both"
        or config.vlm_revision == "main"
        or config.encoder_revision == "main"
    ):
        raise ValueError("Require the original pinned real QA configuration")
    manifest_path = Path(config.manifest_path).resolve()
    manifest = read(manifest_path)
    baseline_dir = Path(baseline_dir or source.parent / "videomme_dev_frozen")
    baseline_protocol = read(baseline_dir / "protocol.json")
    if report_digest(baseline_protocol) != protocol[
        "baseline_protocol_sha256"
    ] or baseline_protocol["manifest_sha256"] != report_digest(manifest):
        raise ValueError("Original baseline protocol/manifest changed")
    items = [v for v in manifest["videos"] if v["video_id"] == video_id]
    if (
        len(items) != 1
        or items[0]["split"] != "dev"
        or video_id not in protocol["video_ids"]
    ):
        raise ValueError("Select one development video in the source run")
    item = items[0]
    questions = [
        q for q in item["questions"] if q["question_id"] == question_id
    ]
    if len(questions) != 1:
        raise ValueError("Select one question in that video")
    annotation = questions[0]
    question = Question(
        question_id,
        annotation["text"],
        tuple(annotation["options"]),
        annotation["answer_index"],
    )
    row = read(
        source
        / "questions"
        / (report_digest([video_id, question_id]) + ".json")
    )
    if row["video_id"] != video_id or row["question_id"] != question_id:
        raise ValueError("Source question identity mismatch")
    prepared = read(source / "plans" / (report_digest(video_id) + ".json"))
    if prepared["identity"] != {
        "video_id": video_id,
        "protocol_sha256": report_digest(protocol),
        "candidate_sha256": row["candidate_sha256"],
    }:
        raise ValueError("Source plan identity mismatch")
    plan = prepared["questions"][question_id]["D"]
    if any(row["conditions"]["D"][k] != value for k, value in plan.items()):
        raise ValueError("Saved D plan/question mismatch")
    validate_plan(plan, config.frame_budget, "question_relevance")
    strict_trials(row["conditions"]["D"]["trials"], question, config)
    metadata = read(source / "backend.json")
    if report_digest(metadata) != protocol["backend_metadata_sha256"]:
        raise ValueError("Original backend metadata identity mismatch")
    if not (manifest_path.parent / item["path"]).is_file():
        raise ValueError("Source video is missing")
    event_id, selections = variants(plan, remove_index, replacements)
    locked = {
        "version": 1,
        "result_kind": "posthoc_human_evidence_single_frame_replacement",
        "source_dir": str(source),
        "source_protocol_sha256": report_digest(protocol),
        "source_question_sha256": report_digest(row),
        "manifest_sha256": report_digest(manifest),
        "backend_metadata_sha256": report_digest(metadata),
        "candidate_sha256": row["candidate_sha256"],
        "config": protocol["config"],
        "video_id": video_id,
        "question_id": question_id,
        "event_id": event_id,
        "removed_index": remove_index,
        "replacement_indices": list(replacements),
        "conditions": selections,
        "orders": [t["order"] for t in row["conditions"]["D"]["trials"]],
        "code_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                *FROZEN_CODE,
                "frame_replacement_probe.py",
                "midpoint_review.py",
                "bottleneck_diagnostics.py",
                "cache.py",
                "qa_diagnostics.py",
                "types.py",
            )
        },
        "interpretation": (
            "One human-selected dev question. Each treatment replaces one "
            "frame, never adds frames. Orders are paired repetitions, not "
            "six questions. No population CI or automatic-selector claim. "
            "Gold and "
            "evidence notes never enter the QA prompt."
        ),
    }
    return config, manifest_path, item, question, row, metadata, locked


def summarize(conditions, question):
    reference = [t["order"] for t in conditions["rerun_D"]["trials"]]
    for condition in conditions.values():
        if [t["order"] for t in condition["trials"]] != reference:
            raise ValueError("Mismatched paired option orders")
    accuracy = {
        name: sum(t["scoring_correct"] for t in c["trials"]) / len(reference)
        for name, c in conditions.items()
    }
    paired = []
    for number, order in enumerate(reference):
        entry = {"order_number": number + 1, "order": order}
        for name, condition in conditions.items():
            trial = condition["trials"][number]
            scores = trial["scoring"]["option_scores"]
            correct_position = order.index(question.answer_index)
            entry[name + "_original_answer_index"] = trial[
                "scored_original_index"
            ]
            entry[name + "_correct"] = trial["scoring_correct"]
            entry[name + "_gold_margin"] = scores[correct_position] - max(
                s for i, s in enumerate(scores) if i != correct_position
            )
        paired.append(entry)
    old, rerun = (
        conditions[name]["trials"] for name in ("saved_D", "rerun_D")
    )
    return {
        "n_questions": 1,
        "n_videos": 1,
        "n_orders": len(reference),
        "order_mean_accuracy": accuracy,
        "delta_vs_rerun_D_pp": {
            name: 100 * (value - accuracy["rerun_D"])
            for name, value in accuracy.items()
            if name.startswith("replace_")
        },
        "baseline_reproduction": {
            "same_original_predictions": all(
                a["scored_original_index"] == b["scored_original_index"]
                for a, b in zip(old, rerun)
            ),
            "max_abs_option_score_difference": max(
                abs(x - y)
                for a, b in zip(old, rerun)
                for x, y in zip(
                    a["scoring"]["option_scores"],
                    b["scoring"]["option_scores"],
                )
            ),
        },
        "paired_orders": paired,
    }


def run(
    source,
    output,
    video_id="856",
    question_id="856-1",
    remove_index=1152,
    replacements=(1035,),
    resume=False,
    check_inputs=False,
    baseline_dir=None,
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Use an output outside the original experiment")
    config, manifest_path, item, question, row, metadata, protocol = (
        load_inputs(
            source,
            video_id,
            question_id,
            remove_index,
            replacements,
            baseline_dir,
        )
    )
    if check_inputs:
        return {
            "status": "validated_without_decode_or_inference",
            "video_id": video_id,
            "question_id": question_id,
            "conditions": protocol["conditions"],
            "planned_qa_calls": len(protocol["conditions"])
            * len(protocol["orders"]),
        }
    if resume:
        if read(output / "protocol.json") != protocol:
            raise ValueError("Resume protocol changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", protocol)
    print(
        f"Decoding {video_id} and checking original candidate pixels",
        flush=True,
    )
    frames, times, duration = load_candidates(item, manifest_path, config)
    video = Video(
        video_id,
        "dev",
        frames,
        times,
        duration,
        (question,),
        item["source_id"],
    )
    validate_video(video, config)
    if array_digest(frames, times) != row["candidate_sha256"]:
        raise ValueError(
            "Candidate pixels/timestamps differ from original run"
        )
    saved = row["conditions"]["D"]
    if not np.array_equal(
        times[saved["selected_indices"]], saved["selected_timestamps"]
    ):
        raise ValueError("Original selected timestamps differ")
    backend = None

    def compute(indices, order):
        nonlocal backend
        if backend is None:
            seed_call(config)
            backend = build_backend(config)
            if backend.metadata() != metadata:
                raise ValueError(
                    "Model/runtime metadata differs from original run"
                )
            write_json(output / "backend.json", metadata)
        return measured_score(backend, video, question, indices, order, config)

    conditions = {
        "saved_D": {
            "selected_indices": saved["selected_indices"],
            "selected_timestamps": saved["selected_timestamps"],
            "trials": saved["trials"],
            "origin": "original experiment",
        }
    }
    completed, digest = 0, report_digest(protocol)
    for name, indices in protocol["conditions"].items():
        trials = []
        for number, order in enumerate(protocol["orders"]):
            identity = {
                "protocol_sha256": digest,
                "indices": indices,
                "order": order,
            }
            trial, reused = cached_trial(
                output / "trials" / (report_digest(identity) + ".json"),
                identity,
                lambda: compute(indices, order),
            )
            trials.append(trial)
            completed += 1
            write_json(
                output / "progress.json",
                {
                    "completed_calls": completed,
                    "total_calls": len(protocol["conditions"])
                    * len(protocol["orders"]),
                    "status": "running",
                },
            )
            print(
                f"{name}: order {number + 1}/{len(protocol['orders'])} "
                f"correct={trial['scoring_correct']} reused={reused}",
                flush=True,
            )
        strict_trials(trials, question, config)
        conditions[name] = {
            "selected_indices": indices,
            "selected_timestamps": times[indices].tolist(),
            "trials": trials,
            "origin": "probe QA",
        }
    summary = summarize(conditions, question)
    result = {
        "protocol_sha256": digest,
        "question": question.text,
        "options": list(question.options),
        "answer_index_evaluator_only": question.answer_index,
        "conditions": conditions,
        "summary": summary,
        "interpretation": protocol["interpretation"],
    }
    write_json(output / "results.json", result)
    write_csv(output / "paired_orders.csv", summary["paired_orders"])
    lines = [
        "# 단일 프레임 교체 진단",
        "",
        question.text,
        "",
        f"문항 {video_id}/{question_id}: 후보 {remove_index} 한 장만 교체.",
        "나머지 입력·장수·QA·선택지 순서는 고정. "
        "사람이 근거를 지정한 사후 진단.",
        "",
        "| 조건 | 순서 평균 정확도 | 재실행 D 대비 (%p) |",
        "|---|---:|---:|",
    ]
    for name, accuracy in summary["order_mean_accuracy"].items():
        delta = summary["delta_vs_rerun_D_pp"].get(name)
        lines.append(
            f"| {name} | {accuracy:.2%} | "
            f"{format(delta, '+.2f') if delta is not None else '—'} |"
        )
    baseline = summary["baseline_reproduction"]
    lines += [
        "",
        f"원 D 예측 재현: {baseline['same_original_predictions']}",
        "원 D와 재실행의 선택지 점수 최대 절대 차이: "
        f"{baseline['max_abs_option_score_difference']:.8g}",
        "",
        "6개 순서는 한 문항의 반복이며 독립 문항이 아닙니다. "
        "이 결과로 자동 프레임 선택기의 일반적 성능을 주장할 수 없습니다.",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "progress.json",
        {
            "completed_calls": completed,
            "total_calls": completed,
            "status": "complete",
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/videomme_event_factorial"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/frame_replace_856_1_1035"),
    )
    parser.add_argument("--video-id", default="856")
    parser.add_argument("--question-id", default="856-1")
    parser.add_argument("--remove-index", type=int, default=1152)
    parser.add_argument("--replacements", type=int, nargs="+", default=[1035])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.input_dir,
        args.output_dir,
        args.video_id,
        args.question_id,
        args.remove_index,
        args.replacements,
        args.resume,
        args.check_inputs,
        args.baseline_dir,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
