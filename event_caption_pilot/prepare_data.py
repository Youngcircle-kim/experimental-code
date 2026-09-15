"""Prepare three official real-video examples for a local inference check."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .cache import write_json
from .config import DEFAULT_SEED
from .reproducibility import fix_seed

INITIAL_SEED_METADATA = fix_seed()
LOGGER = logging.getLogger(__name__)
ANNOTATION_URL = (
    "https://raw.githubusercontent.com/ptchallenge-workshop/"
    "ptchallenge-workshop.github.io/main/data.json"
)
VIDEO_BASE_URL = (
    "https://storage.googleapis.com/dm-perception-test/visualisation_videos"
)
DATASET_URL = "https://github.com/google-deepmind/perception_test"
EXPLORER_URL = "https://ptchallenge-workshop.github.io/explore.html"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ALLOWED_VIDEO_IDS = frozenset({"video_7766", "video_1726", "video_5698"})
SPLIT_NOTE = (
    "Custom video-disjoint smoke-test split of publicly exposed Perception "
    "Test examples; not the official benchmark train/validation/test split. "
    "This subset checks real inference execution, not model performance."
)


@dataclass(frozen=True)
class PreparationConfig:
    """Keep download limits and explicit smoke-example choices serializable."""

    output_dir: str = "data/perception_smoke"
    seed: int = DEFAULT_SEED
    video_selections: tuple[tuple[str, str, tuple[int, ...]], ...] = (
        ("video_7766", "dev", (0, 4)),
        ("video_1726", "eval", (10, 12)),
        ("video_5698", "eval", (3, 5)),
    )
    max_download_bytes: int = 200_000_000
    chunk_bytes: int = 1_048_576
    request_timeout_seconds: float = 60.0
    expected_checksums: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        """Reject unsupported IDs, ambiguous splits and invalid limits.

        Returns:
            None.
        """
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a nonempty string")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("max_download_bytes", "chunk_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_bytes > self.max_download_bytes:
            raise ValueError("chunk_bytes must not exceed max_download_bytes")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive")
        seen: set[str] = set()
        splits: set[str] = set()
        for video_id, split, question_ids in self.video_selections:
            if video_id not in ALLOWED_VIDEO_IDS or video_id in seen:
                raise ValueError(
                    "Use each of the three allowed video IDs once"
                )
            if split not in {"dev", "eval"}:
                raise ValueError("split must be dev or eval")
            if not question_ids or any(
                type(question_id) is not int or question_id < 0
                for question_id in question_ids
            ):
                raise ValueError("Question IDs must be nonnegative integers")
            if len(set(question_ids)) != len(question_ids):
                raise ValueError("Question selections cannot contain repeats")
            seen.add(video_id)
            splits.add(split)
        if seen != ALLOWED_VIDEO_IDS or splits != {"dev", "eval"}:
            raise ValueError(
                "Select the three fixed videos across dev and eval"
            )
        checksum_names: set[str] = set()
        for name, checksum in self.expected_checksums:
            if name not in ALLOWED_VIDEO_IDS | {"annotations"}:
                raise ValueError(f"Unknown checksum resource: {name}")
            if name in checksum_names:
                raise ValueError(f"Duplicate checksum resource: {name}")
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError(
                    "Expected SHA256 must be 64 lowercase hex digits"
                )
            checksum_names.add(name)


def file_sha256(path: Path, chunk_bytes: int) -> tuple[str, int]:
    """Hash an existing file without loading the video into memory.

    Args:
        path: Local source file.
        chunk_bytes: Maximum bytes read in each iteration.

    Returns:
        SHA256 hexadecimal digest and actual file length.
    """
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def download_resource(
    url: str,
    destination: Path,
    config: PreparationConfig,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Stream one bounded download and retain verifiable original provenance.

    Existing resources must match their saved provenance and optional expected
    checksum. Untracked files are not overwritten or silently adopted.

    Args:
        url: Official resource URL.
        destination: Local filename, with a separate provenance sidecar.
        config: Download size, chunk and timeout limits.
        expected_sha256: Optional independently provided expected checksum.

    Returns:
        Original source, checksum, byte count and download timestamp record.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_name(destination.name + ".provenance.json")
    if destination.exists():
        if not destination.is_file() or not sidecar.is_file():
            raise ValueError(
                f"Existing resource lacks provenance: {destination}; "
                "choose an empty output directory"
            )
        with sidecar.open(encoding="utf-8") as stream:
            stored: dict[str, Any] = json.load(stream)
        checksum, size_bytes = file_sha256(destination, config.chunk_bytes)
        if (
            stored.get("source_url") != url
            or stored.get("sha256") != checksum
            or stored.get("size_bytes") != size_bytes
            or not 0 < size_bytes <= config.max_download_bytes
            or (expected_sha256 is not None and checksum != expected_sha256)
        ):
            raise ValueError(
                f"Existing resource failed verification: {destination}"
            )
        LOGGER.info("Verified existing resource: %s", destination)
        return stored
    if sidecar.exists():
        raise ValueError(
            f"Orphan resource provenance requires review: {sidecar}"
        )
    LOGGER.info("Downloading %s", url)
    temporary_path: Path | None = None
    try:
        request = Request(url, headers={"Accept-Encoding": "identity"})
        with urlopen(
            request, timeout=config.request_timeout_seconds
        ) as response:
            expected_size = response.headers.get("Content-Length")
            if expected_size is not None:
                expected_size = int(expected_size)
                if not 0 < expected_size <= config.max_download_bytes:
                    raise ValueError("Remote resource exceeds download limit")
            digest = hashlib.sha256()
            size_bytes = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".part",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                while chunk := response.read(config.chunk_bytes):
                    size_bytes += len(chunk)
                    if size_bytes > config.max_download_bytes:
                        raise ValueError(
                            "Download exceeded max_download_bytes"
                        )
                    digest.update(chunk)
                    output.write(chunk)
            if size_bytes == 0 or (
                expected_size is not None and size_bytes != expected_size
            ):
                raise ValueError("Downloaded resource is empty or truncated")
            checksum = digest.hexdigest()
            if expected_sha256 is not None and checksum != expected_sha256:
                raise ValueError("Downloaded resource failed expected SHA256")
            stored = {
                "source_url": url,
                "resolved_url": response.geturl(),
                "sha256": checksum,
                "size_bytes": size_bytes,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "license": "CC-BY-4.0",
                "license_url": LICENSE_URL,
                "attribution": (
                    "DeepMind Technologies Limited; Perception Test"
                ),
            }
        assert temporary_path is not None and size_bytes > 0
        if destination.exists() or sidecar.exists():
            raise ValueError(
                "Resource appeared during download; refusing overwrite"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        write_json(sidecar, stored)
        LOGGER.info("Downloaded %s (%d bytes)", destination, size_bytes)
        return stored
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def select_questions(
    annotations: dict[str, Any], video_id: str, question_ids: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Preserve author question wording, options and correct-option indices.

    Args:
        annotations: Official dataset explorer JSON.
        video_id: Selected original Perception Test video identifier.
        question_ids: Explicit original question IDs in desired output order.

    Returns:
        Validated pipeline question rows without invented evidence intervals.
    """
    original_rows = annotations.get(video_id)
    if not isinstance(original_rows, list) or not original_rows:
        raise ValueError(f"No author annotations found for {video_id}")
    indexed: dict[int, dict[str, Any]] = {}
    for row in original_rows:
        if not isinstance(row, dict) or type(row.get("id")) is not int:
            raise ValueError("Author question rows need integer IDs")
        if row["id"] in indexed:
            raise ValueError(
                "Author annotation IDs are unexpectedly duplicated"
            )
        indexed[row["id"]] = row
    selected: list[dict[str, Any]] = []
    for question_id in question_ids:
        if question_id not in indexed:
            raise ValueError(
                f"Missing author question {video_id}/{question_id}"
            )
        row = indexed[question_id]
        text = row.get("question")
        options = row.get("options")
        answer_index = row.get("answer_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Author question text must not be empty")
        if (
            not isinstance(options, list)
            or len(options) < 2
            or any(
                not isinstance(option, str) or not option.strip()
                for option in options
            )
        ):
            raise ValueError("Author options must be nonempty strings")
        if type(answer_index) is not int or not 0 <= answer_index < len(
            options
        ):
            raise ValueError("Author answer_id does not identify an option")
        selected.append(
            {
                "question_id": str(question_id),
                "text": text,
                "options": options.copy(),
                "answer_index": answer_index,
                "evidence_intervals": [],
                "annotation_source": "official_perception_test_explorer",
                "source_question_id": question_id,
            }
        )
    assert len(selected) == len(question_ids)
    assert all(row["evidence_intervals"] == [] for row in selected)
    return selected


def preserve_or_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a generated artifact without replacing a different existing file.

    Args:
        path: Manifest or preparation provenance destination.
        payload: Deterministic JSON-serializable artifact.

    Returns:
        None.
    """
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        canonical = json.loads(json.dumps(payload, allow_nan=False))
        if previous != canonical:
            raise ValueError(
                f"Existing artifact differs: {path}; "
                "choose another output directory"
            )
        return
    write_json(path, payload)


def prepare_dataset(config: PreparationConfig) -> dict[str, Any]:
    """Download the explicit real-video subset and export a reusable manifest.

    Args:
        config: Validated preparation settings.

    Returns:
        Preparation provenance including the absolute manifest path.
    """
    config.validate()
    fix_seed(config.seed)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_checksums = dict(config.expected_checksums)
    annotation_path = output_dir / "official_annotations.json"
    annotation_provenance = download_resource(
        ANNOTATION_URL,
        annotation_path,
        config,
        expected_checksums.get("annotations"),
    )
    with annotation_path.open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    if not isinstance(annotations, dict):
        raise ValueError("Official annotations must be a JSON object")
    # Validate every selected author question before downloading video bytes.
    questions = {
        video_id: select_questions(annotations, video_id, question_ids)
        for video_id, _, question_ids in config.video_selections
    }
    resources: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for video_id, split, _ in config.video_selections:
        relative_path = Path("videos") / f"{video_id}.mp4"
        resources[video_id] = download_resource(
            f"{VIDEO_BASE_URL}/{video_id}.mp4",
            output_dir / relative_path,
            config,
            expected_checksums.get(video_id),
        )
        manifest_rows.append(
            {
                "video_id": video_id,
                "source_id": f"perception_test:{video_id}",
                "split": split,
                "path": relative_path.as_posix(),
                "timestamp_mode": "constant_fps",
                "questions": questions[video_id],
            }
        )
    manifest = {
        "dataset": "Perception Test public explorer smoke subset",
        "split_note": SPLIT_NOTE,
        "videos": manifest_rows,
    }
    manifest_path = output_dir / "manifest.json"
    provenance = {
        "schema_version": 1,
        "dataset_url": DATASET_URL,
        "explorer_url": EXPLORER_URL,
        "license": "CC-BY-4.0",
        "license_url": LICENSE_URL,
        "attribution": "DeepMind Technologies Limited; Perception Test",
        "split_note": SPLIT_NOTE,
        "timestamp_note": (
            "Manifest requests frame-index/FPS timestamps from official "
            "visualisation MP4 exports; verify FPS during decoding. "
            "No QA evidence intervals are supplied by explorer annotations."
        ),
        "config": asdict(config),
        "manifest_path": str(manifest_path),
        "annotations": annotation_provenance,
        "videos": resources,
        "selected_question_counts": {
            video_id: len(rows) for video_id, rows in questions.items()
        },
    }
    preserve_or_write_json(manifest_path, manifest)
    preserve_or_write_json(output_dir / "preparation.json", provenance)
    LOGGER.info("Prepared real-video smoke manifest: %s", manifest_path)
    return provenance


def main(argv: list[str] | None = None) -> int:
    """Parse preparation settings and download only the three fixed examples.

    Args:
        argv: Optional command-line arguments, otherwise process arguments.

    Returns:
        Zero after successful preparation; one after a logged failure.
    """
    defaults = PreparationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--max-download-bytes", type=int, default=defaults.max_download_bytes
    )
    parser.add_argument(
        "--chunk-bytes", type=int, default=defaults.chunk_bytes
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=defaults.request_timeout_seconds,
    )
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        metavar="RESOURCE=HEX",
        help="Optional checksum for annotations or one selected video ID",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        checksums = tuple(
            tuple(value.split("=", maxsplit=1))
            for value in arguments.expected_sha256
        )
        if any(len(pair) != 2 for pair in checksums):
            raise ValueError("Use --expected-sha256 RESOURCE=HEX")
        config = PreparationConfig(
            output_dir=arguments.output_dir,
            seed=arguments.seed,
            max_download_bytes=arguments.max_download_bytes,
            chunk_bytes=arguments.chunk_bytes,
            request_timeout_seconds=arguments.request_timeout_seconds,
            expected_checksums=checksums,
        )
        prepare_dataset(config)
    except (OSError, ValueError, TypeError) as error:
        LOGGER.error("Preparation failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
