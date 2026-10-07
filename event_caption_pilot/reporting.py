"""Export original boundary frames and manual research review fields."""

from __future__ import annotations

import base64
import csv
import html
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .cache import write_json
from .config import Config
from .types import Video


def png_data_url(frame: np.ndarray) -> str:
    """Encode an RGB frame losslessly using only the standard library.

    Args:
        frame: uint8 RGB array of shape (H, W, 3).

    Returns:
        Inline PNG URL for an offline diagnostic HTML report.
    """
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB frame")
    height, width = frame.shape[:2]
    chunks: list[bytes] = [b"\x89PNG\r\n\x1a\n"]
    raw = b"".join(b"\x00" + row.tobytes() for row in frame)
    for kind, data in (
        (b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        (b"IDAT", zlib.compress(raw)),
        (b"IEND", b""),
    ):
        chunks.append(
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )
    return "data:image/png;base64," + base64.b64encode(
        b"".join(chunks)
    ).decode("ascii")


def export_artifacts(
    report: dict[str, Any],
    videos: list[Video],
    config: Config,
    output_path: Path,
) -> None:
    """Write one complete JSON plus tabular and visual inspection companions.

    Args:
        report: Experiment metadata, config, predictions, metrics and costs.
        videos: Original candidate frames for boundary inspection.
        config: Active report settings.
        output_path: Unique directory for this run.

    Returns:
        None.
    """
    write_json(output_path / "results.json", report)
    flat_fields = [
        "video_id",
        "question_id",
        "question",
        "answer_index",
        *[f"{name}_correct" for name in report["conditions"]],
        "vt_change",
        "question_wall_seconds",
    ]
    with (output_path / "questions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=flat_fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(report["questions"])
    summary = [
        "# VQA ablation results", "", report["result_kind"], "",
        "All accuracy differences below are percentage points. "
        "Intervals use paired video-cluster bootstrap and are not "
        "adjusted for multiple comparisons.", "",
        "| Condition | Accuracy (%) |", "| --- | ---: |",
    ]
    for name, value in report["metrics"].get("accuracies", {}).items():
        summary.append(f"| {name} | {100 * value:.2f} |")
    for hypothesis, comparisons in report["metrics"].get(
        "hypotheses", {}
    ).items():
        summary.extend([
            "", f"## {hypothesis}", "",
            "| Comparison | Difference (pp) | CI (pp) |",
            "| --- | ---: | --- |",
        ])
        for name, result in comparisons.items():
            interval = result["confidence_interval"]
            ci = (
                f"[{100 * interval[0]:.2f}, {100 * interval[1]:.2f}]"
                if interval is not None else "unavailable"
            )
            difference = 100 * result["mean_difference"]
            summary.append(f"| {name} | {difference:.2f} | {ci} |")
    (output_path / "summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    manual_reviews = {
        "instructions": (
            "Fill from original video/frame inspection; "
            "null means unreviewed."
        ),
        "boundary_reviews": [],
        "caption_reviews": [],
    }
    parts = [
        "<!doctype html><html lang='ko'><meta charset='utf-8'>",
        "<title>Event caption pilot diagnostics</title>",
        "<style>body{font:16px system-ui;max-width:1200px;"
        "margin:36px auto;padding:16px}",
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #aaa;padding:8px;text-align:left}",
        ".timeline{display:flex;margin:12px 0}"
        ".event{background:#deebf7;border-right:2px solid #356c9b;",
        "padding:6px 0;font-size:12px;overflow:hidden}"
        "img{max-width:160px;max-height:120px}",
        ".pairs{display:flex;gap:18px;flex-wrap:wrap}"
        ".pair{border:1px solid #ddd;padding:8px}",
        "pre{white-space:pre-wrap}details{margin:12px 0}"
        "h2{margin-top:40px}</style><body>",
        "<h1>Event caption 최소 파일럿</h1>",
        f"<p>실행 모드: <strong>{html.escape(config.mode)}</strong>. "
        "합성 데모의 정확도는 실제 영상 성능 근거가 아닙니다. "
        "경계와 caption 오류는 원본 영상을 보고 판정하세요.</p>",
        "<p>전체 설정·지표·시간·선택 timestamp: "
        "<a href='results.json'>results.json</a>. "
        "수동 검토 입력란: "
        "<a href='manual_review.json'>manual_review.json</a>.</p>",
    ]
    source_videos = {video.video_id: video for video in videos}
    for video_result in report["videos"]:
        video = source_videos[video_result["video_id"]]
        parts.append(
            f"<h2>{html.escape(video.video_id)} "
            f"({html.escape(video.split)})</h2>"
        )
        for diagnostic in video_result["detector_comparisons"]:
            label = (
                f"{diagnostic['detector']} / "
                f"stride={diagnostic['observation_stride']}"
            )
            parts.append(
                f"<details><summary>{label}: "
                f"{len(diagnostic['events'])}개 구간</summary>"
                "<div class='timeline'>"
            )
            for event in diagnostic["events"]:
                width = (
                    100 * event["duration_seconds"] / video.duration_seconds
                )
                parts.append(
                    f"<div class='event' style='width:{width:.5f}%' "
                    f"title='{event['start_seconds']:.3f}–"
                    f"{event['end_seconds']:.3f}s'>"
                    f"{event['duration_seconds']:.2f}s</div>"
                )
            parts.append("</div><div class='pairs'>")
            for boundary in diagnostic["boundaries"][
                : config.diagnostic_max_boundaries
            ]:
                before, after = (
                    boundary["before_index"],
                    boundary["after_index"],
                )
                parts.append(
                    "<div class='pair'>"
                    f"<p>{video.timestamps[before]:.3f}s → "
                    f"{video.timestamps[after]:.3f}s</p>"
                    "<img alt='boundary before' "
                    f"src='{png_data_url(video.frames[before])}'> "
                    "<img alt='boundary after' "
                    f"src='{png_data_url(video.frames[after])}'></div>"
                )
            parts.append(
                "</div><p>경계 미리보기 최대 "
                f"{config.diagnostic_max_boundaries}개; "
                "전체는 JSON 참조.</p></details>"
            )
            manual_reviews["boundary_reviews"].append(
                {
                    "video_id": video.video_id,
                    "detector": diagnostic["detector"],
                    "observation_stride": diagnostic["observation_stride"],
                    "over_segmentation": None,
                    "under_segmentation": None,
                    "sampling_miss": None,
                    "same_background_action_miss": None,
                    "notes": "",
                }
            )
        if video_result["captions"]:
            parts.append(
                "<table><tr><th>고정 event</th>"
                "<th>관측 timestamp (s)</th><th>Caption</th></tr>"
            )
            for event_id, caption in enumerate(video_result["captions"]):
                provenance = caption["provenance"]
                parts.append(
                    f"<tr><td>{event_id}: "
                    f"{provenance['event_start_seconds']:.3f}–"
                    f"{provenance['event_end_seconds']:.3f}</td>"
                    f"<td>{html.escape(str(provenance['input_timestamps']))}</td>"
                    f"<td>{html.escape(caption['text'])}</td></tr>"
                )
                manual_reviews["caption_reviews"].append(
                    {
                        "video_id": video.video_id,
                        "event_id": event_id,
                        "cache_key": caption["cache_key"],
                        "omission": None,
                        "hallucination": None,
                        "action_order_error": None,
                        "notes": "",
                    }
                )
            parts.append("</table>")
    parts.append("</body></html>")
    (output_path / "diagnostics.html").write_text(
        "\n".join(parts), encoding="utf-8"
    )
    write_json(output_path / "manual_review.json", manual_reviews)
