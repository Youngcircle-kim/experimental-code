"""Export verified candidate frames for human midpoint-evidence review."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from .algorithms import sample_indices
from .bottleneck_diagnostics import read, score, validate_plan, write_csv
from .config import Config
from .data import array_digest, decode_video, validate_video
from .event_factorial_dev import validate_trials
from .midpoint_review_html import render_html
from .qa_diagnostics import report_digest
from .reporting import png_data_url
from .types import Question, Video


def choose_rows(
    rows, limit, duration="long", video_id=None, question_id=None, control="C"
):
    """Purposeful, outcome-balanced cases; never a prevalence estimate."""
    if limit < 1:
        raise ValueError("limit must be positive")
    buckets = {
        name: []
        for name in ("improved", "degraded", "both_wrong", "unchanged")
    }
    seen = set()
    for row in rows:
        key = (row["video_id"], row["question_id"])
        if key in seen:
            raise ValueError("Duplicate question")
        seen.add(key)
        if duration != "all" and row["duration_group"] != duration:
            continue
        if video_id is not None and row["video_id"] != video_id:
            continue
        if question_id is not None and row["question_id"] != question_id:
            continue
        d, c = (score(row["conditions"][arm]) for arm in ("D", control))
        delta = d - c
        bucket = (
            "improved"
            if delta > 1e-12
            else "degraded"
            if delta < -1e-12
            else "both_wrong"
            if d == c == 0
            else "unchanged"
        )
        buckets[bucket].append(row)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda r: (
                -abs(
                    score(r["conditions"]["D"])
                    - score(r["conditions"][control])
                ),
                r["video_id"],
                r["question_id"],
            )
        )
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
    if not selected:
        raise ValueError("No questions match filters")
    return selected


def event_records(plan, preview_count):
    if preview_count < 2:
        raise ValueError("preview-count must be at least 2")
    records = []
    for event_id, ((start, stop), allocation) in enumerate(
        zip(plan["events"], plan["allocation"])
    ):
        selected = [i for i in plan["selected_indices"] if start <= i < stop]
        previews = sorted(
            set(selected)
            | set(
                sample_indices(
                    start, stop, min(preview_count, stop - start)
                ).tolist()
            )
        )
        records.append(
            {
                "event_id": event_id,
                "start": start,
                "stop": stop,
                "allocation": allocation,
                "selected_indices": selected,
                "preview_indices": previews,
                "is_single_midpoint": allocation == 1
                and selected == [start + (stop - start - 1) // 2],
            }
        )
    return records


def load_candidates(item, manifest_path, config):
    path = (manifest_path.parent / item["path"]).resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return (
                archive["frames"].copy(),
                archive["timestamps"].astype(float),
                float(archive["duration_seconds"].item()),
            )
    if item.get("timestamp_mode") != "constant_fps":
        raise ValueError("Require constant_fps video or verified NPZ")
    return decode_video(path, config)


def export_review(
    source,
    output,
    limit=12,
    duration="long",
    video_id=None,
    question_id=None,
    control="C",
    preview_count=7,
    all_candidates=False,
    list_only=False,
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the experiment directory")
    if not list_only and output.exists():
        raise FileExistsError("Use a new output directory to preserve reviews")
    if control not in ("B", "C"):
        raise ValueError("control must be B or C")
    if preview_count < 2:
        raise ValueError("preview-count must be at least 2")
    protocol = read(source / "protocol.json")
    settings = dict(protocol["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    rows = choose_rows(
        [read(path) for path in sorted((source / "questions").glob("*.json"))],
        limit,
        duration,
        video_id,
        question_id,
        control,
    )
    if list_only:
        return [
            {
                "video_id": r["video_id"],
                "question_id": r["question_id"],
                "duration_group": r["duration_group"],
                "delta_pp": 100
                * (
                    score(r["conditions"]["D"])
                    - score(r["conditions"][control])
                ),
                "candidate_count": r["conditions"]["D"]["events"][-1][1],
            }
            for r in rows
        ]
    manifest_path = Path(config.manifest_path).resolve()
    manifest = read(manifest_path)
    metadata = {item["video_id"]: item for item in manifest["videos"]}
    plans = {}
    for path in sorted((source / "plans").glob("*.json")):
        p = read(path)
        vid = p["identity"]["video_id"]
        if vid in plans:
            raise ValueError("Duplicate video plan")
        plans[vid] = p
    report = {
        "version": 1,
        "source": str(source),
        "protocol_sha256": report_digest(protocol),
        "manifest_sha256_at_review": report_digest(manifest),
        "questions": [],
    }
    report["selection_note"] = (
        "개선·악화·양쪽 모두 오답·동일 점수를 순환 선택한 사례 검토입니다. "
        "전체 실패율 추정용 표본이 아닙니다. 미리보기의 부재는 근거 부재가 "
        "아닙니다. 구간 판단과 전체 입력의 답변 충분성을 구분하세요."
    )
    prepared = []
    for row in rows:
        vid, qid = row["video_id"], row["question_id"]
        if "video_ids" in protocol and vid not in protocol["video_ids"]:
            raise ValueError("Question video outside saved run")
        item = metadata[vid]
        if item.get("split") != "dev":
            raise ValueError("This tool reviews dev questions only")
        question = next(
            q for q in item["questions"] if q["question_id"] == qid
        )
        q = Question(
            qid,
            question["text"],
            tuple(question["options"]),
            question["answer_index"],
        )
        for arm in ("D", control):
            condition = row["conditions"][arm]
            validate_trials(condition["trials"], q, config)
            validate_plan(
                condition,
                config.frame_budget,
                "candidate_count" if arm == "B" else "question_relevance",
            )
        identity = plans[vid]["identity"]
        if (
            identity["protocol_sha256"] != report["protocol_sha256"]
            or identity["candidate_sha256"] != row["candidate_sha256"]
        ):
            raise ValueError("Plan identity mismatch")
        saved = plans[vid]["questions"][qid]["D"]
        if any(
            row["conditions"]["D"][k] != value for k, value in saved.items()
        ):
            raise ValueError("Saved D plan differs from question checkpoint")
        prepared.append((row, q, event_records(saved, preview_count)))
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    for number, vid in enumerate(sorted({r["video_id"] for r in rows})):
        cases = [(r, q, e) for r, q, e in prepared if r["video_id"] == vid]
        print(
            f"Decoding video {vid}; verifying original candidate hash",
            flush=True,
        )
        frames, times, seconds = load_candidates(
            metadata[vid], manifest_path, config
        )
        video = Video(
            vid,
            "dev",
            frames,
            times,
            seconds,
            tuple(q for _, q, _ in cases),
            metadata[vid]["source_id"],
        )
        validate_video(video, config)
        if (
            array_digest(frames, times)
            != plans[vid]["identity"]["candidate_sha256"]
        ):
            raise ValueError(f"Decoded candidates differ from run: {vid}")
        wanted = (
            set(range(len(frames)))
            if all_candidates
            else {
                i
                for _, _, events in cases
                for event in events
                for i in event["preview_indices"]
            }
        )
        gallery = {}
        for i in sorted(wanted):
            path = f"frames/v{number}_{i:05d}.png"
            (output / path).write_bytes(
                base64.b64decode(png_data_url(frames[i]).split(",", 1)[1])
            )
            gallery[str(i)] = {"src": path, "timestamp": float(times[i])}
        for row, q, events in cases:
            plan = row["conditions"]["D"]
            if plan["events"][-1][1] != len(frames) or not np.array_equal(
                times[plan["selected_indices"]], plan["selected_timestamps"]
            ):
                raise ValueError("Saved candidate indices/timestamps differ")
            report["questions"].append(
                {
                    "video_id": vid,
                    "question_id": q.question_id,
                    "text": q.text,
                    "options": list(q.options),
                    "answer_index": q.answer_index,
                    "duration_group": row["duration_group"],
                    "comparison": "D_minus_" + control,
                    "delta_pp": 100
                    * (score(plan) - score(row["conditions"][control])),
                    "accuracy_D": score(plan),
                    "accuracy_control": score(row["conditions"][control]),
                    "candidate_count": len(frames),
                    "candidate_sha256": row["candidate_sha256"],
                    "all_candidates_exported": all_candidates,
                    "frames": gallery,
                    "selected_indices": plan["selected_indices"],
                    "events": events,
                }
            )
        del frames, times, video
        print(
            f"Exported video {vid}: {len(gallery)} candidate PNGs", flush=True
        )
    # Keep the purposefully sampled question order in the user interface.
    order = {(r["video_id"], r["question_id"]): i for i, r in enumerate(rows)}
    report["questions"].sort(
        key=lambda q: order[(q["video_id"], q["question_id"])]
    )
    report["report_id"] = report_digest(report)
    (output / "review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / "index.html").write_text(render_html(report), encoding="utf-8")
    return report


def assess(report, annotations):
    """Summarize human judgments without inferring evidence automatically."""
    original = dict(report)
    report_id = original.pop("report_id")
    if (
        report_digest(original) != report_id
        or annotations.get("report_id") != report_id
    ):
        raise ValueError("Review identity mismatch")
    lookup = {
        (q["video_id"], q["question_id"]): q for q in report["questions"]
    }
    seen, results = set(), []
    for reviewed in annotations["questions"]:
        key = (reviewed["video_id"], reviewed["question_id"])
        if key in seen or key not in lookup:
            raise ValueError("Unknown/duplicate review question")
        seen.add(key)
        q = lookup[key]
        sufficient = reviewed["selected_input_sufficient"]
        if sufficient not in ("unknown", "yes", "no"):
            raise ValueError("Invalid overall sufficiency judgment")
        events = {e["event_id"]: e for e in q["events"]}
        seen_events = set()
        for judgment in reviewed["events"]:
            eid = judgment["event_id"]
            if type(eid) is not int or eid in seen_events or eid not in events:
                raise ValueError("Unknown/duplicate review event")
            seen_events.add(eid)
            event = events[eid]
            midpoint = judgment["midpoint_sufficient"]
            extra = judgment["extra_frames_add_evidence"]
            if midpoint not in ("unknown", "yes", "no", "na") or extra not in (
                "unknown",
                "yes",
                "no",
            ):
                raise ValueError("Invalid event judgment")
            if not event["is_single_midpoint"] and midpoint not in (
                "na",
                "unknown",
            ):
                raise ValueError(
                    "Midpoint judgment requires exactly one midpoint"
                )
            indices = judgment["evidence_indices"]
            if (
                not isinstance(indices, list)
                or any(type(i) is not int for i in indices)
                or len(indices) != len(set(indices))
                or any(
                    not event["start"] <= i < event["stop"]
                    or str(i) not in q["frames"]
                    for i in indices
                )
            ):
                raise ValueError(
                    "Evidence indices must be exported event candidates"
                )
            missing = sorted(set(indices) - set(event["selected_indices"]))
            if extra == "yes" and not missing:
                raise ValueError(
                    "Additional evidence needs an unselected witness index"
                )
            status = "human_review_pending"
            if (
                event["is_single_midpoint"]
                and midpoint == "no"
                and extra == "yes"
            ):
                status = "human_identified_midpoint_miss"
            elif event["allocation"] == 0 and extra == "yes":
                status = "human_identified_omitted_event_evidence"
            elif event["is_single_midpoint"] and midpoint == "yes":
                status = "human_judged_midpoint_sufficient_for_event"
            elif midpoint == "no":
                status = "midpoint_insufficient_alternative_unconfirmed"
            results.append(
                {
                    "video_id": key[0],
                    "question_id": key[1],
                    "event_id": eid,
                    "allocation": event["allocation"],
                    "status": status,
                    "selected_input_sufficient": sufficient,
                    "midpoint_sufficient": midpoint,
                    "extra_frames_add_evidence": extra,
                    "evidence_indices": indices,
                    "unselected_evidence_indices": missing,
                    "notes": judgment.get("notes", ""),
                }
            )
        if seen_events != set(events):
            raise ValueError(
                "Include every event, marking unreviewed as unknown"
            )
    if seen != set(lookup):
        raise ValueError(
            "Include every question, marking unreviewed as unknown"
        )
    counts = {
        status: sum(r["status"] == status for r in results)
        for status in sorted({r["status"] for r in results})
    }
    return {
        "report_id": report_id,
        "n_questions": len(seen),
        "event_status_counts": counts,
        "events": results,
        "note": (
            "Human judgments on selected cases; not benchmark accuracy "
            "or population failure rates."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/videomme_event_factorial"),
    )
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--limit", type=int, default=12)
    export.add_argument(
        "--duration",
        choices=["long", "medium", "short", "all"],
        default="long",
    )
    export.add_argument("--video-id")
    export.add_argument("--question-id")
    export.add_argument("--control", choices=["B", "C"], default="C")
    export.add_argument("--preview-count", type=int, default=7)
    export.add_argument("--all-candidates", action="store_true")
    export.add_argument("--list-only", action="store_true")
    review = commands.add_parser("assess")
    review.add_argument("--report", type=Path, required=True)
    review.add_argument("--annotations", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export_review(
            args.input_dir,
            args.output_dir,
            args.limit,
            args.duration,
            args.video_id,
            args.question_id,
            args.control,
            args.preview_count,
            args.all_candidates,
            args.list_only,
        )
        print(
            json.dumps(
                result
                if args.list_only
                else {
                    "questions": len(result["questions"]),
                    "html": str(args.output_dir.resolve() / "index.html"),
                },
                ensure_ascii=False,
            )
        )
    else:
        result = assess(read(args.report), read(args.annotations))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "assessment.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_csv(args.output_dir / "assessment.csv", result["events"])
        print(json.dumps(result["event_status_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
