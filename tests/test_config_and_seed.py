"""Protect reproducible settings and fail-fast experiment validation."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

import numpy as np

from event_caption_pilot.config import Config
from event_caption_pilot.reproducibility import fix_seed


class ConfigAndSeedTests(unittest.TestCase):
    """Check user-facing configuration and both mandatory random generators."""

    def test_seed_replays_python_and_numpy(self) -> None:
        """Verify identical seeds replay both global generators.

        Returns:
            None.
        """
        fix_seed(seed=173)
        first_python = [random.random() for _ in range(4)]
        first_numpy = np.random.normal(size=(3, 4))
        fix_seed(seed=173)
        self.assertEqual(first_python, [random.random() for _ in range(4)])
        np.testing.assert_array_equal(
            first_numpy, np.random.normal(size=(3, 4))
        )
        fix_seed(seed=174)
        self.assertNotEqual(first_python, [random.random() for _ in range(4)])

    def test_default_configuration_is_valid(self) -> None:
        """Verify the immediately runnable default configuration.

        Returns:
            None.
        """
        config = Config()
        config.validate()
        self.assertEqual(config.to_dict()["seed"], config.seed)
        self.assertEqual(config.to_dict()["frame_budget"], config.frame_budget)

    def test_invalid_counts_and_nonfinite_values_are_rejected(self) -> None:
        """Reject settings before invalid ranges or array operations execute.

        Returns:
            None.
        """
        invalid_changes = (
            {"frame_budget": 0},
            {"frame_budget": 2.5},
            {"frame_budget": True},
            {"seed": -1},
            {"seed": 2.5},
            {"seed": True},
            {"allocation_temperature": float("nan")},
            {"boundary_threshold": float("inf")},
            {"text_weight": float("nan")},
            {"observation_strides": (0, 1)},
            {"confidence_level": 1.0},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(Config(), **changes).validate()

    def test_real_mode_needs_explicit_manifest_and_backend(self) -> None:
        """Prevent a demo backend from masquerading as a real experiment.

        Returns:
            None.
        """
        with self.assertRaises(ValueError):
            replace(Config(), mode="real").validate()
        with self.assertRaises(ValueError):
            replace(
                Config(), mode="real", manifest_path="videos.json"
            ).validate()
