"""Create a reproducible custom Video-MME split from official flat rows."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .cache import write_json
from .config import Config


def convert_rows(rows, video_root):
    videos = {}
    sources = {}
    for row in rows:
        identifier, source = str(row["video_id"]), row["videoID"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source):
            raise ValueError("Unsafe videoID")
        if source in sources and sources[source] != identifier:
            raise ValueError("Same source has multiple video IDs")
        sources[source] = identifier
        entry = videos.setdefault(
            identifier,
            {
                "video_id": identifier,
                "source_id": "videomme:" + source,
                "path": str((Path(video_root) / (source + ".mp4")).resolve()),
                "timestamp_mode": "constant_fps",
                "duration": row["duration"],
                "domain": row["domain"],
                "sub_category": row["sub_category"],
                "questions": [],
            },
        )
        if (
            entry["source_id"] != "videomme:" + source
            or entry["duration"] != row["duration"]
            or entry["domain"] != row["domain"]
        ):
            raise ValueError("Inconsistent video metadata")
        options = []
        if len(row["options"]) != 4 or row["answer"] not in tuple("ABCD"):
            raise ValueError("Expected official four-option annotations")
        for i, text in enumerate(row["options"]):
            match = re.fullmatch(r"([A-D])\.\s*(.+)", text.strip())
            if not match or match.group(1) != "ABCD"[i]:
                raise ValueError("Option labels do not match ABCD order")
            options.append(match.group(2))
        qid = str(row["question_id"])
        if any(q["question_id"] == qid for q in entry["questions"]):
            raise ValueError("Duplicate question")
        entry["questions"].append(
            {
                "question_id": qid,
                "text": row["question"],
                "options": options,
                "answer_index": "ABCD".index(row["answer"]),
                "task_type": row["task_type"],
            }
        )
    return sorted(videos.values(), key=lambda v: v["video_id"])


def stratified_split(videos, dev_count, eval_count, seed):
    if min(dev_count, eval_count) < 1 or dev_count + eval_count > len(videos):
        raise ValueError("Not enough videos for requested disjoint splits")
    strata = defaultdict(list)
    for video in sorted(videos, key=lambda v: v["video_id"]):
        strata[(video["duration"], video["domain"])].append(video)
    rng = random.Random(seed)
    keys = sorted(strata)
    rng.shuffle(keys)
    for pool in strata.values():
        rng.shuffle(pool)
    selected = []
    for split, count in (("dev", dev_count), ("eval", eval_count)):
        n = 0
        while n < count:
            for key in keys:
                if strata[key] and n < count:
                    selected.append({**strata[key].pop(), "split": split})
                    n += 1
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Official flat JSON rows; otherwise fetch HF annotations",
    )
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-videos", type=int, default=60)
    parser.add_argument("--eval-videos", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    provenance = {}
    if args.annotations:
        rows = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
        provenance["annotation_path"] = str(args.annotations.resolve())
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download

        repo = "lmms-lab/Video-MME"
        info = HfApi().dataset_info(repo)
        files = sorted(
            s.rfilename
            for s in info.siblings
            if s.rfilename.startswith("videomme/test-")
            and s.rfilename.endswith(".parquet")
        )
        if not files:
            raise ValueError("Official annotation parquet files not found")
        rows = []
        for name in files:
            path = hf_hub_download(
                repo, name, repo_type="dataset", revision=info.sha
            )
            rows.extend(pq.read_table(path).to_pylist())
        provenance.update(repo=repo, revision=info.sha, files=files)
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()
    selected = stratified_split(
        convert_rows(rows, args.video_root),
        args.dev_videos,
        args.eval_videos,
        args.seed,
    )
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "dataset": "Video-MME custom video-disjoint dev/holdout",
            "official_benchmark_result": False,
            "videos": selected,
        },
    )
    settings = json.loads(args.base_config.read_text(encoding="utf-8-sig"))
    settings.update(
        manifest_path=str(manifest_path.resolve()),
        candidate_fps=2.0,
        max_candidate_frames=8192,
        frame_budget=16,
        seed=args.seed,
    )
    settings["observation_strides"] = tuple(
        settings.get("observation_strides", [1, 2, 4])
    )
    config = Config(**settings)
    config.validate()
    if config.vlm_revision == "main":
        raise ValueError("Use a pinned base config")
    write_json(args.output_dir / "config.json", config.to_dict())
    write_json(
        args.output_dir / "question_types.json",
        {
            f"{v['video_id']}/{q['question_id']}": q["task_type"]
            for v in selected
            for q in v["questions"]
        },
    )
    report = {
        "seed": args.seed,
        "annotation_sha256": digest,
        "provenance": provenance,
        "selection": "round-robin duration/domain; seeded within strata; "
        "independent of file availability, answers and model scores",
        "videos_by_split": dict(Counter(v["split"] for v in selected)),
        "questions_by_split": dict(
            Counter(v["split"] for v in selected for q in v["questions"])
        ),
        "strata": dict(
            Counter(
                f"{v['split']}/{v['duration']}/{v['domain']}" for v in selected
            )
        ),
        "missing_videos": [
            v["path"] for v in selected if not Path(v["path"]).is_file()
        ],
    }
    write_json(args.output_dir / "preparation.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("videos_by_split", "questions_by_split")},
            indent=2,
        )
    )
    print(f"Missing selected videos: {len(report['missing_videos'])}")


if __name__ == "__main__":
    main()
