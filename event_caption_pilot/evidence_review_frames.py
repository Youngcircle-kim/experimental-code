"""CPU-only contact sheets for blinded AI review of fixed video cases.

Saved QA inputs remain exact Linux-decoded pixels. Newly decoded Windows
frames carry separate provenance; they are not asserted to reproduce Linux
pixels. All candidate/native mappings are verified and decoding is sequential.
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .data import array_digest
from .portable_budget import file_identity, read_json, save_json
from .qa_diagnostics import report_digest
from .video_sampling import plan_native_candidates

DECODER = "local Windows OpenCV sequential grab/retrieve; CPU only"


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", value
    ):
        raise ValueError("Unsafe case or set identifier")
    return value


def _font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _case_id(case):
    return _safe_id(case["case_id"])


def _times(case):
    times = np.asarray(case["candidate_timestamps"], dtype=np.float64)
    if (
        times.shape != (case["candidate_count"],)
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Invalid frozen candidate timestamps")
    return times


def _indices(values, count):
    if not values or any(type(i) is not int for i in values):
        raise ValueError(
            "Require a nonempty list of integer candidate indices"
        )
    if (
        len(values) != len(set(values))
        or min(values) < 0
        or max(values) >= count
    ):
        raise ValueError("Duplicate or out-of-range candidate index")
    return sorted(values)


def overview_indices(times, cosines, uniform_count=64, bins=32):
    """Uniform timeline plus highest query cosine in each temporal bin."""
    times = np.asarray(times, dtype=np.float64)
    scores = np.asarray(cosines, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or scores.shape != times.shape
        or not np.isfinite(times).all()
        or not np.isfinite(scores).all()
        or np.any(np.diff(times) <= 0)
        or uniform_count < 1
        or bins < 1
    ):
        raise ValueError("Invalid overview timestamps, scores or counts")
    uniform = sorted(
        set(
            np.linspace(
                0,
                len(times) - 1,
                min(uniform_count, len(times)),
                dtype=int,
            ).tolist()
        )
    )
    boundaries = np.linspace(
        times[0], np.nextafter(times[-1], np.inf), bins + 1
    )
    best = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        candidates = np.flatnonzero((times >= start) & (times < stop))
        if len(candidates):
            # np.argmax picks the earliest candidate when scores tie.
            best.append(int(candidates[np.argmax(scores[candidates])]))
    return {"uniform64": uniform, "query_bin32": sorted(best)}


def exact_pixels(case):
    """Read and verify immutable source QA pixels without local decoding."""
    info = case["qa_pixel_cache"]
    path = Path(info["npz_path"])
    metadata = read_json(Path(info["metadata_path"]))
    if (
        file_identity(path)["sha256"] != info["file_sha256"]
        or metadata["file_sha256"] != info["file_sha256"]
        or metadata["array_sha256"] != info["array_sha256"]
        or metadata["source_sha256"] != case["source_video"]["sha256"]
        or metadata["video_id"] != case["video_id"]
        or metadata["qa_side"] != 224
    ):
        raise ValueError("Frozen QA pixel cache identity mismatch")
    with np.load(path, allow_pickle=False) as saved:
        indices = saved["indices"]
        frames = saved["frames"]
        times = saved["timestamps"]
    if (
        indices.dtype != np.int64
        or frames.dtype != np.uint8
        or times.dtype != np.float64
        or frames.shape != (len(indices), 224, 224, 3)
        or indices.tolist() != sorted(set(indices.tolist()))
        or indices.tolist() != metadata["indices"]
        or not np.array_equal(times, _times(case)[indices])
        or array_digest(indices, frames, times) != info["array_sha256"]
    ):
        raise ValueError("Frozen QA pixel cache content mismatch")
    return dict(zip(indices.tolist(), frames)), metadata


def contact_sheets(case, label, indices, image_paths, output):
    """Compose 4x4 unmodified 224-pixel tiles with candidate/time labels."""
    output.mkdir(parents=True, exist_ok=True)
    times = _times(case)
    question = case["question"]
    header = [
        f"{_case_id(case)} | video {case['video_id']} | "
        f"question {case['question_id']} | {label}",
        *textwrap.wrap(question["text"], width=108),
    ]
    for i, option in enumerate(question["options"]):
        header.extend(textwrap.wrap(f"{i + 1}. {option}", width=108))
    header.append("Blinded AI review; no gold answer or QA outcome shown.")
    header_height = 16 * len(header) + 16
    sheets = []
    for page, offset in enumerate(range(0, len(indices), 16), start=1):
        selected = indices[offset : offset + 16]
        canvas = Image.new("RGB", (896, header_height + 4 * 252), "white")
        draw = ImageDraw.Draw(canvas)
        for line, text in enumerate(header):
            draw.text((8, 8 + line * 16), text, fill="black", font=_font(13))
        for slot, candidate in enumerate(selected):
            x, y = (slot % 4) * 224, header_height + (slot // 4) * 252
            with Image.open(image_paths[candidate]) as tile:
                if tile.size != (224, 224):
                    raise ValueError("Contact sheets require exact 224 tiles")
                canvas.paste(tile.convert("RGB"), (x, y))
            stamp = float(times[candidate])
            minutes, seconds = divmod(stamp, 60)
            draw.text(
                (x + 4, y + 226),
                f"#{candidate} | {int(minutes):02}:{seconds:06.3f}",
                fill="black",
                font=_font(15),
            )
        path = output / f"{label}_{page:02}.png"
        canvas.save(path, compress_level=1)
        sheets.append(str(path.resolve()))
    return sheets


def export_exact(case, root):
    """Export all blinded source sets directly from the verified QA cache."""
    output = root / _case_id(case)
    output.mkdir(parents=True, exist_ok=True)
    frames, metadata = exact_pixels(case)
    times = _times(case)
    exported, sets = {}, {}
    for name, selection in case["sets"].items():
        _safe_id(name)
        indices = _indices(selection["selected_indices"], len(times))
        if len(indices) != 32 or any(i not in frames for i in indices):
            raise ValueError("Source review set must contain 32 cached frames")
        if (
            selection["selected_timestamps"] != times[indices].tolist()
            or array_digest(
                np.stack([frames[i] for i in indices]), times[indices]
            )
            != selection["qa_pixels_sha256"]
        ):
            raise ValueError("Blinded set does not match original QA inputs")
        for i in indices:
            if i not in exported:
                path = output / "exact224" / f"candidate_{i:05}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frames[i]).save(path, compress_level=1)
                exported[i] = path
        sets[name] = {
            "indices": indices,
            "sheets": contact_sheets(
                case,
                name,
                indices,
                exported,
                output / "sheets_exact",
            ),
            "qa_pixels_sha256": selection["qa_pixels_sha256"],
        }
    result = {
        "version": 1,
        "case_id": _case_id(case),
        "case_sha256": report_digest(case),
        "pixel_source": "verified original Linux QA pixel cache",
        "local_video_decoding_performed": False,
        "source_candidate_sha256": metadata["candidate_sha256"],
        "sets": sets,
        "frames": {str(i): str(p.resolve()) for i, p in exported.items()},
    }
    save_json(output / "exact_export.json", result)
    return result


def native_mapping(case, config, fps, native_count, metadata):
    """Reconstruct candidate ordinal -> exact sequential native ordinal."""
    source = metadata["decode_info"]
    sampling = source["candidate_sampling"]
    if (
        fps != source["source_fps"]
        or native_count != sampling["native_frame_count"]
    ):
        raise ValueError("Local FPS/native frame count differs from source")
    native, info = plan_native_candidates(config, fps, native_count)
    times = np.asarray([i / fps for i in native], dtype=np.float64)
    if len(native) != case["candidate_count"] or not np.array_equal(
        times, _times(case)
    ):
        raise ValueError("Native candidate index/timestamp mapping changed")
    return native, info


def export_local(case, config, root, target_indices=None, progress=None):
    """Decode requested native ordinals without seeking or GPU operations."""
    import cv2

    started = perf_counter()
    output = root / _case_id(case)
    output.mkdir(parents=True, exist_ok=True)
    old_frames, old_metadata = exact_pixels(case)
    times = _times(case)
    groups = overview_indices(times, case["candidate_question_cosines"])
    baseline_indices = {
        i
        for selection in case["sets"].values()
        for i in selection["selected_indices"]
    }
    selected = sorted(
        baseline_indices
        | {i for values in groups.values() for i in values}
        | set(target_indices or [])
    )
    _indices(selected, len(times))
    native_images = {i for values in groups.values() for i in values}
    native_images.update(target_indices or [])
    if target_indices:
        groups["targeted"] = sorted(target_indices)
    path = Path(case["source_video"]["path"])
    identity = file_identity(path)
    if any(
        identity[k] != case["source_video"][k]
        for k in ("sha256", "size_bytes")
    ):
        raise ValueError("Original source video bytes changed")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot open original source video")
    local_frames, entries = {}, {}
    last_progress = started
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        native_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        native, sampling = native_mapping(
            case,
            config,
            fps,
            native_count,
            old_metadata,
        )
        desired = {native[i]: i for i in selected}
        final_native = max(desired)
        for native_index in range(final_native + 1):
            if not cap.grab():
                raise ValueError(
                    f"Decoding ended before native frame {native_index}"
                )
            if native_index in desired:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise ValueError(
                        f"Cannot retrieve native frame {native_index}"
                    )
                candidate = desired[native_index]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                small = cv2.resize(rgb, (224, 224))
                local_frames[candidate] = small
                small_path = (
                    output / "local224" / f"candidate_{candidate:05}.png"
                )
                small_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(small).save(small_path, compress_level=1)
                native_path = None
                if candidate in native_images:
                    native_path = (
                        output / "native" / f"candidate_{candidate:05}.png"
                    )
                    native_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(native_path, compress_level=1)
                entry = {
                    "candidate_index": candidate,
                    "native_frame_index": native_index,
                    "timestamp_seconds": float(times[candidate]),
                    "native_width_height": [
                        int(rgb.shape[1]),
                        int(rgb.shape[0]),
                    ],
                    "local224_path": str(small_path.resolve()),
                    "native_path": str(native_path.resolve())
                    if native_path
                    else None,
                    "local224_array_sha256": array_digest(small),
                    "native_array_sha256": array_digest(rgb),
                    "pixel_source": DECODER,
                }
                if candidate in old_frames:
                    difference = np.abs(
                        small.astype(np.int16)
                        - old_frames[candidate].astype(np.int16)
                    )
                    entry["comparison_to_linux224"] = {
                        "exact_equal": bool(
                            np.array_equal(small, old_frames[candidate])
                        ),
                        "mean_absolute_difference": float(difference.mean()),
                        "max_absolute_difference": int(difference.max()),
                        "changed_channel_fraction": float(
                            np.mean(difference != 0)
                        ),
                    }
                entries[str(candidate)] = entry
            if progress and perf_counter() - last_progress > 5:
                progress(
                    {
                        "stage": "sequential_decode",
                        "native_frame": native_index,
                        "last_requested_native_frame": final_native,
                        "frames_exported": len(local_frames),
                        "frames_requested": len(selected),
                    }
                )
                last_progress = perf_counter()
    finally:
        cap.release()
    indices = np.asarray(selected, dtype=np.int64)
    frames = np.stack([local_frames[i] for i in selected])
    selected_times = times[indices]
    npz_path = output / "local224_union.npz"
    _save_npz(
        npz_path, indices=indices, frames=frames, timestamps=selected_times
    )
    image_paths = {i: Path(entries[str(i)]["local224_path"]) for i in selected}
    sheets = {
        name: contact_sheets(
            case, name, ids, image_paths, output / "sheets_local"
        )
        for name, ids in groups.items()
    }
    comparisons = [
        e["comparison_to_linux224"]
        for e in entries.values()
        if "comparison_to_linux224" in e
    ]
    result = {
        "version": 1,
        "status": "complete",
        "case_id": _case_id(case),
        "video_id": case["video_id"],
        "question_id": case["question_id"],
        "case_sha256": report_digest(case),
        "decoder": DECODER,
        "opencv_version": cv2.__version__,
        "gpu_inference_performed": False,
        "seek_performed": False,
        "sampling": sampling,
        "native_ordinals_checked": final_native + 1,
        "source_video_sha256": identity["sha256"],
        "source_candidate_sha256": old_metadata["candidate_sha256"],
        "local_candidate_pool_sha256": None,
        "pixel_identity_note": (
            "Source bytes and full timestamp/native-index mapping match. "
            "Only requested pixels were decoded; local pixels are not "
            "asserted identical to the Linux source candidate pool."
        ),
        "local_pixel_cache": {
            "npz_path": str(npz_path.resolve()),
            "file_sha256": file_identity(npz_path)["sha256"],
            "array_sha256": array_digest(indices, frames, selected_times),
            "source_video_sha256": identity["sha256"],
            "source_candidate_sha256": old_metadata["candidate_sha256"],
            "decoder": DECODER,
        },
        "indices": selected,
        "frames": entries,
        "overview_groups": groups,
        "sheets": sheets,
        "linux_overlap": {
            "frames_compared": len(comparisons),
            "exact_equal_frames": sum(c["exact_equal"] for c in comparisons),
            "mean_frame_absolute_difference": float(
                np.mean([c["mean_absolute_difference"] for c in comparisons])
            )
            if comparisons
            else None,
        },
        "export_seconds": perf_counter() - started,
    }
    save_json(output / "local_export.json", result)
    return result


def export_probe_cache(case, local_export, conditions, path):
    """Freeze one local-decoder union for paired baseline/repair/sham QA."""
    cache = local_export["local_pixel_cache"]
    source = Path(cache["npz_path"])
    if (
        local_export["case_sha256"] != report_digest(case)
        or file_identity(source)["sha256"] != cache["file_sha256"]
    ):
        raise ValueError("Local evidence export identity changed")
    indices = sorted(
        {i for c in conditions.values() for i in c["selected_indices"]}
    )
    _indices(indices, case["candidate_count"])
    for condition in conditions.values():
        if (
            len(
                _indices(
                    condition["selected_indices"], case["candidate_count"]
                )
            )
            != 32
        ):
            raise ValueError("Every probe condition must contain 32 frames")
    with np.load(source, allow_pickle=False) as saved:
        src_i, src_f, src_t = (
            saved["indices"],
            saved["frames"],
            saved["timestamps"],
        )
    if array_digest(src_i, src_f, src_t) != cache["array_sha256"]:
        raise ValueError("Local evidence pixels changed")
    positions = {int(i): n for n, i in enumerate(src_i)}
    if any(i not in positions for i in indices):
        raise ValueError("Probe needs targeted export for missing candidates")
    selected = [positions[i] for i in indices]
    wanted = np.asarray(indices, dtype=np.int64)
    frames, times = src_f[selected], src_t[selected]
    _save_npz(path, indices=wanted, frames=frames, timestamps=times)
    result = {
        **cache,
        "npz_path": str(path.resolve()),
        "file_sha256": file_identity(path)["sha256"],
        "array_sha256": array_digest(wanted, frames, times),
        "source_local_export_sha256": report_digest(local_export),
        "conditions_sha256": report_digest(conditions),
    }
    save_json(path.with_suffix(".json"), result)
    return result


def run(args):
    manifest = read_json(args.manifest)
    cases = manifest["cases"]
    if len({_case_id(c) for c in cases}) != len(cases):
        raise ValueError("Duplicate blinded case ID")
    if args.case_ids:
        cases = [c for c in cases if c["case_id"] in args.case_ids]
        if {c["case_id"] for c in cases} != set(args.case_ids):
            raise ValueError("Unknown requested blinded case")
    if not cases or args.workers not in (1, 2):
        raise ValueError("Require cases and one or two CPU workers")
    settings = dict(
        manifest.get("candidate_config")
        or read_json(args.source_protocol)["candidate_config"]
    )
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = read_json(args.target_request) if args.target_request else {}
    statuses = {c["case_id"]: {"stage": "pending"} for c in cases}
    lock = threading.Lock()

    def update(case_id, data):
        with lock:
            statuses[case_id].update(data)
            save_json(
                root / f"progress_{args.command}.json",
                {
                    "manifest_sha256": report_digest(manifest),
                    "stage": args.command,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "cases": statuses,
                    "gpu_inference_performed": False,
                },
            )

    def worker(case):
        case_id = case["case_id"]
        update(case_id, {"stage": "starting"})
        try:
            if args.command == "exact":
                result = export_exact(case, root)
            else:
                request = targets.get(case_id, {})
                indices = set(request.get("indices", []))
                for start, stop in request.get("ranges", []):
                    if (
                        type(start) is not int
                        or type(stop) is not int
                        or not 0 <= start < stop <= case["candidate_count"]
                    ):
                        raise ValueError(
                            "Target ranges must be half-open "
                            "candidate intervals"
                        )
                    indices.update(range(start, stop))
                result = export_local(
                    case,
                    config,
                    root,
                    sorted(indices),
                    lambda d: update(case_id, d),
                )
            update(case_id, {"stage": "complete"})
            print(f"Completed {args.command} export: {case_id}", flush=True)
            return result
        except Exception as error:
            update(
                case_id,
                {
                    "stage": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(worker, c) for c in cases]
        for future in as_completed(jobs):
            future.result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("exact", "overview", "target"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-protocol",
        type=Path,
        default=Path(
            "outputs/rtx3060-fixed-events-20261027/full224/protocol.json"
        ),
    )
    parser.add_argument("--target-request", type=Path)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "target" and args.target_request is None:
        parser.error("target requires --target-request")
    run(args)


if __name__ == "__main__":
    main()
