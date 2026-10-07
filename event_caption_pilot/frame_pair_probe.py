"""Test two evidence frames with or without the other selected frames."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .backends import build_backend
from .bottleneck_diagnostics import read, write_csv
from .cache import write_json
from .data import array_digest, validate_video
from .event_factorial_dev import cached_trial, measured_score, validate_indices
from .frame_replacement_probe import load_inputs, strict_trials, summarize
from .midpoint_review import load_candidates
from .order_experiments import seed_call
from .qa_diagnostics import report_digest
from .types import Video


def pair_selection(original, remove_index, pair, candidate_count):
    """Keep N-1 original frames and add two new frames, yielding N+1."""
    if len(pair) != 2 or any(type(i) is not int for i in pair):
        raise ValueError("Require exactly two integer evidence indices")
    if len(set(pair)) != 2 or set(pair) & set(original):
        raise ValueError("Evidence pair must be distinct and unselected")
    if type(remove_index) is not int or original.count(remove_index) != 1:
        raise ValueError("Removed frame must occur exactly once")
    validate_indices(original, candidate_count, len(original))
    selected = sorted([i for i in original if i != remove_index] + list(pair))
    validate_indices(selected, candidate_count, len(original) + 1)
    if set(original) - set(selected) != {remove_index}:
        raise ValueError("Other original frames must remain unchanged")
    return selected


def prepare_pair(
    source,
    video_id,
    question_id,
    remove_index,
    pair,
    baseline_dir=None,
    only_pair=False,
):
    if len(pair) != 2:
        raise ValueError("Require two evidence frames")
    config, manifest_path, item, question, row, metadata, protocol = (
        load_inputs(
            source, video_id, question_id, remove_index, pair, baseline_dir
        )
    )
    # load_inputs verifies each member is an unselected candidate in the same
    # single-midpoint event. Combine them without removing another frame.
    original = protocol["conditions"]["rerun_D"]
    selected = pair_selection(
        original, remove_index, pair, row["conditions"]["D"]["events"][-1][1]
    )
    name = f"replace_{remove_index}_with_pair_{pair[0]}_{pair[1]}"
    if only_pair:
        selected = sorted(pair)
        name = f"replace_all_with_pair_only_{pair[0]}_{pair[1]}"
    protocol["conditions"] = {"rerun_D": original, name: selected}
    protocol["result_kind"] = "posthoc_two_evidence_frames_budget_increased"
    protocol["only_pair"] = only_pair
    if only_pair:
        protocol["result_kind"] = "posthoc_two_evidence_frames_only"
    protocol["frame_counts"] = {
        k: len(v) for k, v in protocol["conditions"].items()
    }
    protocol["budget_matched"] = False
    protocol["code_sha256"][Path(__file__).name] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    protocol["interpretation"] = (
        "One human-selected question. Replace one midpoint with two evidence "
        "frames and retain all other frames. Treatment has one more frame "
        "than control: not a fixed-budget selector comparison. Same QA and "
        "paired option orders. No independent-question CI or population claim."
    )
    if only_pair:
        protocol["interpretation"] = (
            "One human-selected question. Treatment contains only the two "
            "evidence frames; all original selected frames are omitted. "
            "Two frames versus the original control budget; not a "
            "fixed-budget selector comparison. Same QA and paired orders. "
            "No independent-question CI or population claim."
        )
    return config, manifest_path, item, question, row, metadata, protocol


def run(
    source,
    output,
    video_id="856",
    question_id="856-1",
    remove_index=1152,
    pair=(1035, 1036),
    baseline_dir=None,
    resume=False,
    check_inputs=False,
    only_pair=False,
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Use an output outside the source experiment")
    config, manifest_path, item, question, row, metadata, protocol = (
        prepare_pair(
            source,
            video_id,
            question_id,
            remove_index,
            pair,
            baseline_dir,
            only_pair,
        )
    )
    total = len(protocol["conditions"]) * len(protocol["orders"])
    if check_inputs:
        return {
            "status": "validated_without_decode_or_inference",
            "frame_counts": protocol["frame_counts"],
            "budget_matched": False,
            "planned_qa_calls": total,
            "conditions": protocol["conditions"],
        }
    if resume:
        if read(output / "protocol.json") != protocol:
            raise ValueError("Resume protocol changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", protocol)
    print(
        f"Decoding {video_id}; verifying original candidate pixels", flush=True
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
    saved = row["conditions"]["D"]
    if array_digest(frames, times) != row[
        "candidate_sha256"
    ] or not np.array_equal(
        times[saved["selected_indices"]], saved["selected_timestamps"]
    ):
        raise ValueError("Original candidate pixels/timestamps differ")
    backend = None

    def compute(indices, order):
        nonlocal backend
        if backend is None:
            seed_call(config)
            backend = build_backend(config)
            if backend.metadata() != metadata:
                raise ValueError("Model/runtime metadata differs")
            write_json(output / "backend.json", metadata)
        return measured_score(backend, video, question, indices, order, config)

    conditions = {
        "saved_D": {
            "selected_indices": saved["selected_indices"],
            "selected_timestamps": saved["selected_timestamps"],
            "trials": saved["trials"],
        }
    }
    digest, completed = report_digest(protocol), 0
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
                    "total_calls": total,
                    "status": "running",
                },
            )
            print(
                f"{name}: {len(indices)} frames; order {number + 1}/"
                f"{len(protocol['orders'])}; "
                f"correct={trial['scoring_correct']} "
                f"reused={reused}",
                flush=True,
            )
        strict_trials(trials, question, config)
        conditions[name] = {
            "selected_indices": indices,
            "selected_timestamps": times[indices].tolist(),
            "trials": trials,
        }
    summary = summarize(conditions, question)
    summary["frame_counts"] = {
        k: len(v["selected_indices"]) for k, v in conditions.items()
    }
    summary["budget_matched"] = False
    write_json(
        output / "results.json",
        {
            "protocol_sha256": digest,
            "question": question.text,
            "options": list(question.options),
            "answer_index_evaluator_only": question.answer_index,
            "conditions": conditions,
            "summary": summary,
            "interpretation": protocol["interpretation"],
        },
    )
    write_csv(output / "paired_orders.csv", summary["paired_orders"])
    lines = [
        "# 근거 두 장 단독 입력 진단"
        if only_pair
        else "# 근거 두 장 동시 입력 진단",
        "",
        question.text,
        "",
        (
            f"{list(pair)} 두 장만 사용. 기존 D의 나머지 입력은 모두 제외."
            if only_pair
            else f"{remove_index} 한 장을 {list(pair)} 두 장으로 "
            "교체. 나머지 입력 유지."
        ),
        "대조와 처치의 입력 장수가 다른 사후 진단입니다.",
        "",
        "| 조건 | 입력 장수 | 정답 순서 수 | 순서 평균 정확도 |",
        "|---|---:|---:|---:|",
    ]
    for name, condition in conditions.items():
        correct = sum(t["scoring_correct"] for t in condition["trials"])
        lines.append(
            f"| {name} | {len(condition['selected_indices'])} | "
            f"{correct}/{len(condition['trials'])} | "
            f"{summary['order_mean_accuracy'][name]:.2%} |"
        )
    lines += [
        "",
        "원 D 재현 여부: "
        f"{summary['baseline_reproduction']['same_original_predictions']}",
        "선택지 점수 최대 절대 차이: "
        f"{summary['baseline_reproduction']['max_abs_option_score_difference']:.8g}",
        "",
        "선택지 순서는 한 문항의 반복입니다. "
        "사람이 고른 두 근거 프레임의 효과이며 자동 선택 성능이 아닙니다.",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "progress.json",
        {
            "completed_calls": completed,
            "total_calls": total,
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
    )
    parser.add_argument("--video-id", default="856")
    parser.add_argument("--question-id", default="856-1")
    parser.add_argument("--remove-index", type=int, default=1152)
    parser.add_argument("--pair", type=int, nargs=2, default=[1035, 1036])
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--only-pair", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(
            "outputs/frame_pair_only_856_1_1035_1036"
            if args.only_pair
            else "outputs/frame_pair_856_1_1035_1036"
        )
    print(
        json.dumps(
            run(
                args.input_dir,
                args.output_dir,
                args.video_id,
                args.question_id,
                args.remove_index,
                args.pair,
                args.baseline_dir,
                args.resume,
                args.check_inputs,
                args.only_pair,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
