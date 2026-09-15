"""Seed all requested RNGs before constructing data or frozen models."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from .config import DEFAULT_SEED


def fix_seed(
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    seed_torch: bool = False,
    seed_tensorflow: bool = False,
    cublas_workspace_config: str = ":4096:8",
) -> dict[str, Any]:
    """Seed Python, NumPy and explicitly requested deep-learning frameworks.

    Args:
        seed: Global seed; local NumPy Generators must also receive a seed.
        deterministic: Require supported deterministic framework operations.
        seed_torch: Import and configure PyTorch when True.
        seed_tensorflow: Import and configure TensorFlow when True.
        cublas_workspace_config: CUDA BLAS setting applied before torch import.

    Returns:
        Reproducibility metadata, including the startup hash seed limitation.
    """
    random.seed(seed)
    np.random.seed(seed)
    metadata: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "python_hash_seed_at_startup": os.environ.get("PYTHONHASHSEED"),
        "hash_seed_note": (
            "Set PYTHONHASHSEED before process startup; "
            "not changed at runtime."
        ),
    }
    if seed_torch:
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.use_deterministic_algorithms(deterministic)
        metadata["torch_version"] = torch.__version__
        metadata["cuda_version"] = torch.version.cuda
    if seed_tensorflow:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
        metadata["tensorflow_version"] = tf.__version__
    return metadata


INITIAL_SEED_METADATA = fix_seed()
