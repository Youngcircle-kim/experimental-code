"""Extract selected Video-MME videos from local ZIPs with CRC validation."""

import argparse
import json
import zipfile
from pathlib import Path

from .prepare_holdout_videos import (
    contained_target,
    existing_matches,
    extract_entry,
)


def load_targets(manifest_path, video_root, split="dev", duration="long"):
    if split not in {"dev", "eval", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    if duration not in {"short", "medium", "long", "all"}:
        raise ValueError(f"Unsupported duration: {duration}")
    manifest_path = Path(manifest_path).resolve()
    root = Path(video_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = {}
    for video in manifest["videos"]:
        if split != "all" and video["split"] != split:
            continue
        if duration != "all" and video["duration"] != duration:
            continue
        target = Path(video["path"])
        if not target.is_absolute():
            target = manifest_path.parent / target
        target = contained_target(target, root)
        if target.suffix.lower() != ".mp4":
            raise ValueError(f"Require an MP4 video target: {target}")
        name = target.name.casefold()
        if name in targets:
            raise ValueError(f"Duplicate manifest video basename: {name}")
        targets[name] = {
            "video_id": video["video_id"],
            "path": str(target),
            "questions": len(video["questions"]),
        }
    if not targets:
        raise ValueError(f"No {split}/{duration} videos in the manifest")
    return root, targets


def inspect_inputs(
    manifest_path, archive_dir, video_root, split="dev", duration="long"
):
    """Check all selected archive matches before creating any files."""
    root, targets = load_targets(manifest_path, video_root, split, duration)
    archives = sorted(Path(archive_dir).glob("videos_chunked*.zip"))
    matches = {name: [] for name in targets}
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                # Only the manifest chooses destinations, never ZIP paths.
                name = member.filename.replace("\\", "/").rsplit("/", 1)[-1]
                name = name.casefold()
                if not member.is_dir() and name in targets:
                    matches[name].append({
                        "archive": str(archive.resolve()),
                        "member": member.filename,
                        "size_bytes": member.file_size,
                        "crc32": member.CRC,
                    })
    missing = [targets[name]["path"] for name, m in matches.items() if not m]
    duplicates = [name for name, m in matches.items() if len(m) > 1]
    if missing or duplicates:
        raise ValueError(
            f"Archive matches incomplete: missing={missing}; "
            f"duplicate basenames={duplicates}. "
            "Archives are required even for existing MP4s to validate CRC."
        )
    entries = []
    for name, target in targets.items():
        entry = {**target, **matches[name][0]}
        if entry["size_bytes"] <= 0:
            raise ValueError(f"Empty archived video: {entry['member']}")
        entry["already_present"] = existing_matches(
            Path(entry["path"]), entry["size_bytes"], entry["crc32"]
        )
        entries.append(entry)
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "video_root": str(root),
        "split": split,
        "duration": duration,
        "n_videos": len(entries),
        "n_questions": sum(e["questions"] for e in entries),
        "n_archives": len(archives),
        "already_present": sum(e["already_present"] for e in entries),
        "total_size_bytes": sum(e["size_bytes"] for e in entries),
        "entries": entries,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument(
        "--archive-dir", type=Path, default=Path("data/videomme/archives")
    )
    result.add_argument(
        "--video-root", type=Path, default=Path("data/videomme/videos")
    )
    result.add_argument(
        "--split", choices=("dev", "eval", "all"), default="dev"
    )
    result.add_argument(
        "--duration", choices=("long", "all", "short", "medium"),
        default="long",
    )
    result.add_argument("--check-inputs", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = inspect_inputs(
        args.manifest, args.archive_dir, args.video_root,
        args.split, args.duration,
    )
    if not args.check_inputs:
        for i, entry in enumerate(report["entries"], 1):
            entry["action"] = extract_entry(entry, report["video_root"])
            print(
                f"{entry['action']} {i}/{report['n_videos']}: {entry['path']}",
                flush=True,
            )
    report["check_only"] = args.check_inputs
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
