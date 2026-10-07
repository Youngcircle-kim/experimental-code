"""Prepare eval/long videos from existing local archives, without downloads."""

import argparse
import json
import os
import tempfile
import zipfile
import zlib
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


def contained_target(path, root):
    resolved = Path(path).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Video target is outside video root: {path}")
    return resolved


def load_targets(manifest_path, video_root):
    manifest_path = Path(manifest_path).resolve()
    root = Path(video_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = {}
    for video in manifest["videos"]:
        if video["split"] != "eval" or video["duration"] != "long":
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
        raise ValueError("No eval/long videos in the manifest")
    return root, targets


def file_size_crc(path):
    size, crc = 0, 0
    with Path(path).open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
    return size, crc & 0xFFFFFFFF


def existing_matches(target, size, crc):
    if not target.exists():
        return False
    if not target.is_file() or target.stat().st_size != size:
        raise ValueError(f"Existing video size/type differs: {target}")
    if file_size_crc(target) != (size, crc):
        raise ValueError(f"Existing video CRC differs: {target}")
    return True


def inspect_inputs(manifest_path, archive_dir, video_root):
    """Only read metadata and existing files; never create directories."""
    root, targets = load_targets(manifest_path, video_root)
    archives = sorted(Path(archive_dir).glob("videos_chunked_*.zip"))
    matches = {name: [] for name in targets}
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                # Archive directory paths never determine destination paths.
                name = entry.filename.replace("\\", "/").rsplit("/", 1)[-1]
                name = name.casefold()
                if not entry.is_dir() and name in targets:
                    matches[name].append(
                        {
                            "archive": str(archive.resolve()),
                            "member": entry.filename,
                            "size_bytes": entry.file_size,
                            "crc32": entry.CRC,
                        }
                    )
    missing = [targets[n]["path"] for n, m in matches.items() if not m]
    duplicates = [n for n, m in matches.items() if len(m) > 1]
    if missing or duplicates:
        raise ValueError(
            f"Archive matches incomplete: missing={missing}; "
            f"duplicate basenames={duplicates}"
        )
    plan = []
    for name, target in targets.items():
        entry = {**target, **matches[name][0]}
        if entry["size_bytes"] <= 0:
            raise ValueError(f"Empty archived video: {entry['member']}")
        entry["already_present"] = existing_matches(
            Path(entry["path"]), entry["size_bytes"], entry["crc32"]
        )
        plan.append(entry)
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "video_root": str(root),
        "n_videos": len(plan),
        "n_questions": sum(p["questions"] for p in plan),
        "n_archives": len(archives),
        "already_present": sum(p["already_present"] for p in plan),
        "total_size_bytes": sum(p["size_bytes"] for p in plan),
        "entries": plan,
    }


def extract_entry(entry, video_root):
    """Install a CRC-checked sibling temporary file without overwriting."""
    target = contained_target(entry["path"], Path(video_root).resolve())
    expected = (entry["size_bytes"], entry["crc32"])
    if existing_matches(target, *expected):
        return "reused"
    temporary = None
    try:
        with zipfile.ZipFile(entry["archive"]) as archive:
            member = archive.getinfo(entry["member"])
            if (member.file_size, member.CRC) != expected:
                raise ValueError("Archive member changed since input check")
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                archive.open(member) as source,
                tempfile.NamedTemporaryFile(
                    prefix=target.name + ".",
                    suffix=".part",
                    dir=target.parent,
                    delete=False,
                ) as destination,
            ):
                temporary = Path(destination.name)
                size, crc = 0, 0
                while chunk := source.read(CHUNK_BYTES):
                    destination.write(chunk)
                    size += len(chunk)
                    crc = zlib.crc32(chunk, crc)
                destination.flush()
                os.fsync(destination.fileno())
            if (size, crc & 0xFFFFFFFF) != expected:
                raise ValueError("Extracted video size/CRC differs")
        # A hard-link install is atomic and fails if a destination appeared.
        # The temporary file is on the same volume; its link is removed below.
        os.link(temporary, target)
        return "extracted"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/videomme_pilot/manifest.json"),
    )
    result.add_argument(
        "--archive-dir", type=Path, default=Path("data/videomme/archives")
    )
    result.add_argument(
        "--video-root", type=Path, default=Path("data/videomme/videos")
    )
    result.add_argument("--check-inputs", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = inspect_inputs(args.manifest, args.archive_dir, args.video_root)
    if not args.check_inputs:
        for i, entry in enumerate(report["entries"], 1):
            status = extract_entry(entry, report["video_root"])
            entry["action"] = status
            print(
                f"{status} {i}/{report['n_videos']}: {entry['path']}",
                flush=True,
            )
    report["check_only"] = args.check_inputs
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
