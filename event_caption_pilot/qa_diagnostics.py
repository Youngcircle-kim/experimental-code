"""Inspect saved QA runs without model inference or evaluation retuning."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

from .backends import make_demo_dataset
from .cache import write_json
from .config import Config
from .data import array_digest, load_manifest
from .reporting import png_data_url


def report_digest(report):
    return hashlib.sha256(json.dumps(
        report, sort_keys=True, allow_nan=False
    ).encode()).hexdigest()


def assess_review(report, review):
    """Evaluate human evidence groups, never infer readability from scores.

    Each group contains alternative frames for one required piece of evidence.
    A selection covers the evidence only if it hits every required group.
    """
    if review.get("report_sha256") != report_digest(report):
        raise ValueError("Review belongs to a different results file")
    questions = {
        (row["video_id"], row["question_id"]): row
        for row in report["questions"]
    }
    counts = {v["video_id"]: v["candidate_count"] for v in report["videos"]}
    seen = set()
    output = []
    for entry in review["questions"]:
        key = (entry["video_id"], entry["question_id"])
        if key not in questions or key in seen:
            raise ValueError("Unknown or duplicate review question")
        seen.add(key)
        readable = entry["candidate_evidence_readable"]
        mapping = entry["question_video_mapping_verified"]
        if any(x is not None and type(x) is not bool
               for x in (readable, mapping)):
            raise ValueError("Review flags must be true, false or null")
        groups = entry["required_evidence_groups"]
        if not isinstance(groups, list) or any(
            not isinstance(g, list) or not g or any(
                type(i) is not int or not 0 <= i < counts[key[0]] for i in g
            ) for g in groups
        ):
            raise ValueError("Evidence groups require valid candidate indices")
        if readable is True and not groups:
            raise ValueError("Readable evidence requires at least one group")
        if readable is not True and groups:
            raise ValueError("Evidence groups require readability=true")
        row = questions[key]
        for name, condition in row["conditions"].items():
            hits = [bool(set(g) & set(condition["selected_indices"]))
                    for g in groups]
            covered = all(hits) if readable is True else None
            if mapping is False:
                status = "check_question_video_mapping"
            elif mapping is None or readable is None:
                status = "human_review_pending"
            elif readable is False:
                status = "candidate_evidence_not_readable"
            elif not covered:
                status = "selection_misses_required_evidence"
            elif not condition["correct"]:
                status = "inspect_qa_recognition_reasoning_or_scoring"
            else:
                status = "covered_and_correct"
            output.append({
                "video_id": key[0], "question_id": key[1],
                "condition": name, "correct": condition["correct"],
                "evidence_group_hits": hits,
                "all_required_groups_covered": covered, "status": status,
            })
    if seen != set(questions):
        raise ValueError("Review must include every evaluated question")
    return {"report_sha256": report_digest(report), "diagnoses": output}


def export_qa_diagnostics(report, videos, output_dir):
    """Export all evaluated candidates at their actual input resolution."""
    sources = {v.video_id: v for v in videos}
    evaluated = {q["video_id"] for q in report["questions"]}
    for record in report["videos"]:
        if record["video_id"] in evaluated:
            video = sources[record["video_id"]]
            if array_digest(video.frames, video.timestamps) != record[
                "candidate_content_sha256"
            ]:
                raise ValueError("Decoded candidates differ from saved run")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    images = output_dir / "frames"
    images.mkdir()
    galleries = {}
    for number, video_id in enumerate(sorted(evaluated)):
        video = sources[video_id]
        paths = []
        for index, frame in enumerate(video.frames):
            path = f"frames/v{number}_{index:05d}.png"
            (output_dir / path).write_bytes(base64.b64decode(
                png_data_url(frame).split(",", 1)[1]
            ))
            paths.append(path)
        galleries[video_id] = paths
    parts = [
        "<!doctype html><html lang='ko'><meta charset='utf-8'>",
        "<title>QA 근거 진단</title><style>",
        "body{font:16px system-ui;margin:24px}table{border-collapse:collapse}",
        "td,th{border:1px solid #bbb;padding:6px}"
        ".grid{display:flex;flex-wrap:wrap;gap:10px}"
        "figure{margin:0;width:240px}img{max-width:224px}"
        "figcaption{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>",
        "<h1>QA 근거 프레임 진단</h1>"
        "<p>이미지는 실제 후보 입력 해상도입니다. 클릭하면 PNG를 엽니다. "
        "확대해도 원본에서 소실된 글자는 복원되지 않습니다. "
        "정답·영상 연결을 확인한 후 review.json에 판독 결과를 기록하세요. "
        "점수는 확률이 아닌 저장된 선택지 로그우도입니다.</p>",
    ]
    review = {"schema_version": 1, "report_sha256": report_digest(report),
              "questions": []}
    for row in report["questions"]:
        video = sources[row["video_id"]]
        esc = html.escape
        parts.append(f"<h2>{esc(row['video_id'])} / "
                     f"{esc(row['question_id'])}</h2><p>"
                     f"{esc(row['question'])}</p><p>정답: "
                     f"{esc(row['options'][row['answer_index']])}</p>")
        parts.append("<table><tr><th>조건</th><th>예측</th>"
                     "<th>정답</th><th>정답 점수 − 최고 오답 점수</th>"
                     "<th>배분 / 선택 인덱스 / 선택지별 점수</th></tr>")
        for name, c in row["conditions"].items():
            scores = c["option_scores"]
            margin = scores[row["answer_index"]] - max(
                score for i, score in enumerate(scores)
                if i != row["answer_index"]
            )
            details = {"allocation": c["allocation"],
                       "selected_indices": c["selected_indices"],
                       "option_scores": list(zip(row["options"], scores))}
            parts.append(f"<tr><td>{esc(name)}</td><td>"
                         f"{esc(row['options'][c['predicted_index']])}</td>"
                         f"<td>{c['correct']}</td><td>{margin:.4f}</td>"
                         f"<td>{esc(json.dumps(details, ensure_ascii=False))}"
                         "</td></tr>")
        parts.append("</table><details><summary>전체 후보 및 선택 조건 표시"
                     "</summary><div class='grid'>")
        for index, path in enumerate(galleries[video.video_id]):
            selected = [name for name, c in row["conditions"].items()
                        if index in c["selected_indices"]]
            parts.append(f"<figure><a href='{path}'><img loading='lazy' "
                         f"src='{path}' alt='candidate {index}'></a>"
                         f"<figcaption>#{index} / "
                         f"{video.timestamps[index]:.3f}s<br>"
                         f"{esc(', '.join(selected) or '미선택')}</figcaption>"
                         "</figure>")
        parts.append("</div></details>")
        for name in ("control_v", "treatment_t", "treatment_vt"):
            if name not in row["conditions"]:
                continue
            parts.append(f"<details><summary>{name} 선택 프레임"
                         "</summary><div class='grid'>")
            for i in row["conditions"][name]["selected_indices"]:
                path = galleries[video.video_id][i]
                parts.append(f"<figure><a href='{path}'><img loading='lazy' "
                             f"src='{path}' alt='candidate {i}'></a>"
                             f"<figcaption>#{i} / {video.timestamps[i]:.3f}s"
                             "</figcaption></figure>")
            parts.append("</div></details>")
        review["questions"].append({
            "video_id": row["video_id"], "question_id": row["question_id"],
            "question_video_mapping_verified": None,
            "candidate_evidence_readable": None,
            "required_evidence_groups": [], "notes": "",
        })
    parts.append("</html>")
    (output_dir / "qa_review.html").write_text(
        "\n".join(parts), encoding="utf-8"
    )
    write_json(output_dir / "review.json", review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=str)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text(encoding="utf-8-sig"))
    if args.review:
        review = json.loads(args.review.read_text(encoding="utf-8-sig"))
        diagnosis = assess_review(report, review)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "diagnosis.json", diagnosis)
    else:
        settings = dict(report["config"])
        if args.manifest:
            settings["manifest_path"] = args.manifest
        settings["observation_strides"] = tuple(
            settings["observation_strides"]
        )
        config = Config(**settings)
        config.validate()
        videos = (make_demo_dataset(config) if config.mode == "demo"
                  else load_manifest(config))
        export_qa_diagnostics(report, videos, args.output_dir)
    print(f"QA diagnostics saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
