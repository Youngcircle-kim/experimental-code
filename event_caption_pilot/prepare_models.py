"""Download public checkpoints once and pin their resolved revisions."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from .cache import write_json
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
        patterns = ["*.json", "*.txt", "*.model", "*.safetensors"]
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
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model-cache-dir", default=Config().model_cache_dir)
    parser.add_argument("--output-config", type=Path,
                        default=Path("configs/real.mps.smoke.json"))
    parser.add_argument("--manifest", default="data/perception_smoke/manifest.json")
    arguments_parsed = parser.parse_args(arguments)
    config = replace(
        Config(), mode="real", backend="transformers",
        manifest_path=arguments_parsed.manifest, device=arguments_parsed.device,
        model_dtype="float16", seed_torch=True,
        model_cache_dir=arguments_parsed.model_cache_dir,
        output_dir="outputs/real_mps", cache_dir=".cache/captions_real_mps",
        attention_implementation="eager", candidate_fps=1.0,
        frame_budget=16, max_segments=3, uniform_segments=3,
        caption_max_frames=4, caption_max_new_tokens=40,
        observation_strides=(1, 2), encoder_batch_size=8,
        vlm_min_pixels=12544, vlm_max_pixels=50176,
    )
    config.validate()
    models = prepare_models(config)
    pinned = replace(
        config, encoder_revision=models["encoder"]["revision"],
        vlm_revision=models["vlm"]["revision"], local_files_only=True,
    )
    if arguments_parsed.output_config.exists():
        raise FileExistsError(
            f"Preserving existing config: {arguments_parsed.output_config}"
        )
    write_json(arguments_parsed.output_config, pinned.to_dict())
    write_json(Path(config.model_cache_dir) / "download_provenance.json", models)
    LOGGER.info("Pinned offline config: %s", arguments_parsed.output_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
