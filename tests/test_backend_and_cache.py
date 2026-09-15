"""Verify synthetic fixtures and question-independent caption provenance."""

from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from event_caption_pilot.backends import DemoBackend, make_demo_dataset
from event_caption_pilot.cache import event_caption
from event_caption_pilot.config import Config
from event_caption_pilot.reproducibility import fix_seed
from event_caption_pilot.types import CaptionOutput, Event


class CountingCaptionBackend(DemoBackend):
    """Expose caption invocations without accepting labels or question data."""

    def __init__(self, config: Config) -> None:
        """Initialize the deterministic backend and its observation counter.

        Args:
            config: Active synthetic backend settings.
        """
        super().__init__(config)
        self.caption_calls = 0
        self.observed_timestamps: list[np.ndarray] = []

    def caption(
        self, frames: np.ndarray, timestamps: np.ndarray
    ) -> CaptionOutput:
        """Record exclusively the original RGB observations and timestamps.

        Args:
            frames: Original observations, without annotations.
            timestamps: Matching original timestamps, without question data.

        Returns:
            Deterministic caption derived only from pixels.
        """
        self.caption_calls += 1
        self.observed_timestamps.append(timestamps.copy())
        return super().caption(frames, timestamps)


class StochasticCaptionBackend(DemoBackend):
    """Exercise both required global random generators during captioning."""

    def caption(
        self, frames: np.ndarray, timestamps: np.ndarray
    ) -> CaptionOutput:
        """Return both random draws so differing generator state is visible.

        Args:
            frames: Original sampled observations.
            timestamps: Matching original timestamps.

        Returns:
            Synthetic caption containing Python and NumPy random draws.
        """
        return CaptionOutput(
            text=f"red {random.random():.17g} {np.random.random():.17g}"
        )


class BackendAndCacheTests(unittest.TestCase):
    """Protect the visible-data fixture and reusable caption audit records."""

    def test_demo_replays_and_full_observations_answer_visible_properties(
        self,
    ) -> None:
        """Validate fixture labels directly against visible observations.

        Returns:
            None.
        """
        config = replace(Config(), demo_dev_videos=1, demo_eval_videos=1)
        first_videos = make_demo_dataset(config)
        repeated_videos = make_demo_dataset(config)
        backend = DemoBackend(config)
        for first_video, repeated_video in zip(
            first_videos, repeated_videos, strict=True
        ):
            np.testing.assert_array_equal(
                first_video.frames, repeated_video.frames
            )
            self.assertEqual(first_video.questions, repeated_video.questions)
            for question in first_video.questions:
                answer = backend.answer(
                    first_video.frames,
                    first_video.timestamps,
                    question.text,
                    question.options,
                )
                self.assertEqual(answer.predicted_index, question.answer_index)

    def test_warm_cache_preserves_generation_record_and_ignores_labels(
        self,
    ) -> None:
        """Reuse provenance when questions, options, and labels change.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                Config(),
                cache_dir=directory,
                demo_dev_videos=1,
                demo_eval_videos=1,
                demo_frames_per_video=24,
            )
            video = make_demo_dataset(config)[0]
            backend = CountingCaptionBackend(config)
            indices = np.array([0, 1, 3], dtype=np.int64)
            cold = event_caption(video, Event(0, 4), indices, backend, config)
            changed_question = replace(
                video.questions[0],
                text="A completely different question?",
                options=("new option", "other option"),
                answer_index=1,
            )
            changed_video = replace(video, questions=(changed_question,))
            warm = event_caption(
                changed_video, Event(0, 4), indices, backend, config
            )
            self.assertFalse(cold["cache_hit"])
            self.assertTrue(warm["cache_hit"])
            self.assertEqual(backend.caption_calls, 1)
            self.assertEqual(warm["cache_key"], cold["cache_key"])
            self.assertEqual(
                warm["generated_at_utc"], cold["generated_at_utc"]
            )
            self.assertEqual(
                warm["generation_seconds"], cold["generation_seconds"]
            )
            self.assertEqual(warm["provenance"], cold["provenance"])
            self.assertEqual(warm["current_generation_seconds"], 0.0)
            self.assertEqual(warm["current_input_frame_count"], 0)
            np.testing.assert_array_equal(
                backend.observed_timestamps[0], video.timestamps[indices]
            )

    def test_prompt_seed_and_observations_invalidate_caption_cache(
        self,
    ) -> None:
        """Never reuse captions produced under changed generation inputs.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                Config(),
                cache_dir=directory,
                demo_dev_videos=1,
                demo_eval_videos=1,
                demo_frames_per_video=24,
            )
            video = make_demo_dataset(config)[0]
            indices = np.array([0, 1, 3], dtype=np.int64)
            backend = CountingCaptionBackend(config)
            original = event_caption(
                video, Event(0, 4), indices, backend, config
            )
            for changed_config in (
                replace(config, seed=config.seed + 1),
                replace(config, caption_prompt_version="new-version"),
                replace(
                    config, caption_prompt="New fixed caption instructions."
                ),
            ):
                with self.subTest(config=changed_config):
                    changed = event_caption(
                        video, Event(0, 4), indices, backend, changed_config
                    )
                    self.assertFalse(changed["cache_hit"])
                    self.assertNotEqual(
                        changed["cache_key"], original["cache_key"]
                    )
            changed_frames = video.frames.copy()
            changed_frames[0, 0, 0, 0] = 255 - changed_frames[0, 0, 0, 0]
            changed = event_caption(
                replace(video, frames=changed_frames),
                Event(0, 4),
                indices,
                backend,
                config,
            )
            self.assertFalse(changed["cache_hit"])
            self.assertNotEqual(changed["cache_key"], original["cache_key"])
            self.assertEqual(backend.caption_calls, 5)

    def test_partial_cache_hits_cannot_change_stochastic_event_captions(
        self,
    ) -> None:
        """Reproduce event captions when preceding events skip generation.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            cold_config = replace(
                Config(),
                cache_dir=str(Path(directory) / "cold"),
                demo_dev_videos=1,
                demo_eval_videos=1,
                demo_frames_per_video=24,
            )
            partial_config = replace(
                cold_config, cache_dir=str(Path(directory) / "partial")
            )
            video = make_demo_dataset(cold_config)[0]
            backend = StochasticCaptionBackend(cold_config)
            first_event = Event(0, 4)
            second_event = Event(4, 8)
            first_indices = np.array([0, 1, 3], dtype=np.int64)
            second_indices = np.array([4, 5, 7], dtype=np.int64)

            fix_seed(cold_config.seed)
            event_caption(
                video, first_event, first_indices, backend, cold_config
            )
            cold_second = event_caption(
                video, second_event, second_indices, backend, cold_config
            )
            event_caption(
                video, first_event, first_indices, backend, partial_config
            )
            fix_seed(partial_config.seed)
            partial_first = event_caption(
                video, first_event, first_indices, backend, partial_config
            )
            partial_second = event_caption(
                video, second_event, second_indices, backend, partial_config
            )
            self.assertTrue(partial_first["cache_hit"])
            self.assertFalse(partial_second["cache_hit"])
            self.assertEqual(cold_second["text"], partial_second["text"])
            self.assertEqual(
                cold_second["provenance"]["generation_seed"],
                partial_second["provenance"]["generation_seed"],
            )
