"""Download public checkpoints once and pin their resolved revisions."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from .cache import write_json
from .cli import parse_config
from .config import Config

LOGGER = logging.getLogger(__name__)


def prepare_models(config: Config) -> dict[str, Any]:
    """Fetch only inference weights and tokenizer/processor configuration.

    Args:
        config: Model IDs, requested revisions and local cache location.

    Returns:
        Model source revisions, local snapshot paths and downloaded file sizes.
    """
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    models: dict[str, Any] = {}
    for name, model_id, revision in (
        ("encoder", config.encoder_model, config.encoder_revision),
        ("vlm", config.vlm_model, config.vlm_revision),
    ):
        information = api.model_info(model_id, revision=revision)
        filenames = [sibling.rfilename for sibling in information.siblings]
        use_safetensors = any(
            filename.endswith(".safetensors") for filename in filenames
        )
        patterns = ["*.json", "*.txt", "*.model", "*.jinja", "*.safetensors"]
        if not use_safetensors:
            patterns.append("pytorch_model.bin")
        LOGGER.info("Downloading %s at commit %s", model_id, information.sha)
        snapshot = Path(snapshot_download(
            repo_id=model_id, revision=information.sha,
            cache_dir=config.model_cache_dir, allow_patterns=patterns,
            max_workers=2,
        ))
        files = [path for path in snapshot.rglob("*") if path.is_file()]
        models[name] = {
            "model_id": model_id, "revision": information.sha,
            "snapshot_path": str(snapshot.resolve()),
            "total_file_bytes": sum(path.stat().st_size for path in files),
            "source": f"https://huggingface.co/{model_id}",
        }
        LOGGER.info("Ready: %s (%.2f GB)", name,
                    models[name]["total_file_bytes"] / 1e9)
    return models


def main(arguments: list[str] | None = None) -> int:
    """Prepare checkpoints and emit a pinned, offline real-smoke Config.

    Args:
        arguments: Command-line tokens, or None to use process arguments.

    Returns:
        Zero after the model downloads and configuration export succeed.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="Preserve all experiment settings from this JSON")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--output-config", type=Path,
                        default=Path("configs/real.pinned.json"))
    parser.add_argument("--manifest", default=None)
    arguments_parsed = parser.parse_args(arguments)
    if arguments_parsed.output_config.exists():
        raise FileExistsError(
            f"Preserving existing config: {arguments_parsed.output_config}"
        )
    config = (
        parse_config(["--config", str(arguments_parsed.config)])
        if arguments_parsed.config is not None
        else replace(
            Config(), mode="real", backend="transformers",
            manifest_path="data/perception_smoke/manifest.json",
            device="cuda", model_dtype="bfloat16", seed_torch=True,
            output_dir="outputs/real", cache_dir=".cache/event_caption_real",
        )
    )
    overrides = {
        field: value for field, value in (
            ("device", arguments_parsed.device),
            ("manifest_path", arguments_parsed.manifest),
            ("model_cache_dir", arguments_parsed.model_cache_dir),
        ) if value is not None
    }
    config = replace(config, **overrides)
    config.validate()
    models = prepare_models(config)
    pinned = replace(
        config, encoder_revision=models["encoder"]["revision"],
        vlm_revision=models["vlm"]["revision"], local_files_only=True,
    )
    write_json(arguments_parsed.output_config, pinned.to_dict())
    write_json(
        Path(config.model_cache_dir) / "download_provenance.json", models
    )
    LOGGER.info("Pinned offline config: %s", arguments_parsed.output_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
