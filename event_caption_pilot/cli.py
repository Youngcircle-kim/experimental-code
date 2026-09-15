"""Command-line entry point with all overrides routed through Config."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields, replace
from pathlib import Path

from .config import Config
from .pipeline import run_experiment


def parse_config(arguments: list[str] | None = None) -> Config:
    """Load JSON settings, then apply explicitly provided CLI overrides.

    Args:
        arguments: Argument tokens, or None to use process arguments.

    Returns:
        A fully validated configuration; unspecified CLI flags preserve JSON.
    """
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Question-independent event-caption allocation pilot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="JSON Config overrides")
    parser.add_argument("--mode", choices=("demo", "real"), default=None)
    parser.add_argument("--pilot", choices=("a", "both"), default=None)
    parser.add_argument(
        "--backend", default=None, help="demo, transformers, or module:factory"
    )
    parser.add_argument("--manifest", dest="manifest_path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frame-budget", type=int, default=None)
    parser.add_argument("--detector", choices=("D0", "D1", "D2"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--write-config", type=Path, help="Export effective Config and exit"
    )
    namespace = parser.parse_args(arguments)
    config = defaults
    if namespace.config:
        with namespace.config.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            parser.error("Config JSON must contain an object")
        unknown = payload.keys() - {
            parameter.name for parameter in fields(Config)
        }
        if unknown:
            parser.error(f"Unknown Config parameters: {sorted(unknown)}")
        if "observation_strides" in payload:
            payload["observation_strides"] = tuple(
                payload["observation_strides"]
            )
        config = replace(config, **payload)
    overrides = {
        key: value
        for key, value in vars(namespace).items()
        if key not in {"config", "write_config"} and value is not None
    }
    config = replace(config, **overrides)
    config.validate()
    if namespace.write_config:
        from .cache import write_json

        if namespace.write_config.exists():
            parser.error(
                "--write-config destination already exists; choose a new path"
            )
        write_json(namespace.write_config, config.to_dict())
        logging.getLogger(__name__).info(
            "Config written: %s", namespace.write_config
        )
        parser.exit()
    return config


def main(arguments: list[str] | None = None) -> int:
    """Configure structured logging and execute the experiment.

    Args:
        arguments: Optional command-line argument tokens.

    Returns:
        Zero on success, one when validation, input or model execution fails.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = parse_config(arguments)
        run_experiment(config)
    except Exception:
        logging.getLogger(__name__).exception(
            "Pilot failed; no successful result is claimed"
        )
        return 1
    return 0
