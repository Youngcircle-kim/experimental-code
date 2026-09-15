"""Exercise exports, paired budgets, provenance, and model isolation."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from event_caption_pilot.backends import DemoBackend, make_demo_dataset
from event_caption_pilot.config import Config
from event_caption_pilot.pipeline import CONDITION_KEYS, run_experiment
from event_caption_pilot.types import CaptionOutput, QaOutput, Video


def small_config(directory: str) -> Config:
    """Create a complete but inexpensive pilot configuration.

    Args:
        directory: Isolated temporary directory for outputs and cache.

    Returns:
        Settings with independent development and evaluation videos.
    """
    return replace(
        Config(),
        output_dir=str(Path(directory) / "outputs"),
        cache_dir=str(Path(directory) / "cache"),
        demo_dev_videos=1,
        demo_eval_videos=2,
        demo_questions_per_video=2,
        demo_frames_per_video=24,
        demo_event_count=3,
        demo_frame_height=16,
        demo_frame_width=16,
        frame_budget=6,
        bootstrap_samples=50,
    )


class OriginalFrameSpy(DemoBackend):
    """Reject model calls containing more than allowed observations."""

    def __init__(self, config: Config, videos: list[Video]) -> None:
        """Retain original candidates for checking every model input.

        Args:
            config: Active synthetic settings.
            videos: Original candidate pools and evaluation annotations.
        """
        super().__init__(config)
        self.videos = videos
        self.caption_calls = 0
        self.qa_calls = 0

    def caption(
        self, frames: np.ndarray, timestamps: np.ndarray
    ) -> CaptionOutput:
        """Accept only unchanged original frames and matching timestamps.

        Args:
            frames: Caption observations, with no question or label arguments.
            timestamps: Original observation timestamps.

        Returns:
            Pixel-derived caption after verifying original provenance.
        """
        self.caption_calls += 1
        matching_pool = False
        for video in self.videos:
            indices = np.searchsorted(video.timestamps, timestamps)
            if np.any(indices >= len(video.frames)):
                continue
            if np.array_equal(video.timestamps[indices], timestamps):
                matching_pool |= np.array_equal(video.frames[indices], frames)
        if not matching_pool:
            raise AssertionError(
                "Caption input was not an original candidate subset"
            )
        if len(frames) > self.config.caption_max_frames:
            raise AssertionError(
                "Caption observation count exceeded its fixed budget"
            )
        return super().caption(frames, timestamps)

    def answer(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        question: str,
        options: tuple[str, ...],
    ) -> QaOutput:
        """Verify QA receives original frames and allowed QA text.

        Args:
            frames: Selected original frames, with no attached caption content.
            timestamps: Matching original timestamps.
            question: Unmodified input question.
            options: Original answer options, without the correct answer index.

        Returns:
            Pixel-derived answer independent of stored answer labels.
        """
        self.qa_calls += 1
        matching_question = False
        for video in self.videos:
            for original_question in video.questions:
                if (original_question.text, original_question.options) != (
                    question,
                    options,
                ):
                    continue
                indices = np.searchsorted(video.timestamps, timestamps)
                if np.any(indices >= len(video.frames)):
                    continue
                matching_question |= np.array_equal(
                    video.timestamps[indices], timestamps
                ) and np.array_equal(video.frames[indices], frames)
        if not matching_question:
            raise AssertionError(
                "QA text or options were changed or captions appended"
            )
        if len(frames) != self.config.frame_budget:
            raise AssertionError("QA input violated the exact frame budget")
        return super().answer(frames, timestamps, question, options)


class PipelineTests(unittest.TestCase):
    """Verify the complete research workflow with isolated synthetic runs."""

    def test_cold_and_warm_runs_reproduce_outputs_and_preserve_exact_budgets(
        self,
    ) -> None:
        """Compare scientific outputs independently of cache costs and clocks.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            cold = run_experiment(config)
            warm = run_experiment(config)
            self.assertEqual(
                cold["reproducible_result_sha256"],
                warm["reproducible_result_sha256"],
            )
            self.assertEqual(cold["metrics"], warm["metrics"])
            self.assertEqual(cold["calibration"], warm["calibration"])
            self.assertNotEqual(cold["artifact_dir"], warm["artifact_dir"])
            for filename in (
                "results.json",
                "questions.csv",
                "config.json",
                "diagnostics.html",
                "manual_review.json",
            ):
                self.assertTrue(
                    (Path(warm["artifact_dir"]) / filename).is_file()
                )
            with (Path(warm["artifact_dir"]) / "results.json").open(
                encoding="utf-8"
            ) as stream:
                persisted = json.load(stream)
            self.assertEqual(persisted["metrics"], warm["metrics"])
            self.assertEqual(
                persisted["reproducible_result_sha256"],
                warm["reproducible_result_sha256"],
            )
            for cold_video, warm_video in zip(
                cold["videos"], warm["videos"], strict=True
            ):
                candidates = np.asarray(warm_video["candidate_timestamps"])
                for cold_caption, warm_caption in zip(
                    cold_video["captions"], warm_video["captions"], strict=True
                ):
                    self.assertFalse(cold_caption["cache_hit"])
                    self.assertTrue(warm_caption["cache_hit"])
                    self.assertEqual(
                        cold_caption["generated_at_utc"],
                        warm_caption["generated_at_utc"],
                    )
                    self.assertEqual(
                        cold_caption["provenance"], warm_caption["provenance"]
                    )
                    provenance = warm_caption["provenance"]
                    np.testing.assert_array_equal(
                        candidates[provenance["input_indices"]],
                        provenance["input_timestamps"],
                    )
                self.assertEqual(
                    warm_video["caption_cost"]["current_input_frame_count"], 0
                )
            candidates_by_video = {
                video["video_id"]: np.asarray(video["candidate_timestamps"])
                for video in warm["videos"]
            }
            for row in warm["questions"]:
                for condition in CONDITION_KEYS:
                    result = row["conditions"][condition]
                    indices = result["selected_indices"]
                    self.assertEqual(
                        sum(result["allocation"]), config.frame_budget
                    )
                    self.assertEqual(len(set(indices)), config.frame_budget)
                    self.assertEqual(indices, sorted(indices))
                    np.testing.assert_array_equal(
                        candidates_by_video[row["video_id"]][indices],
                        result["selected_timestamps"],
                    )

    def test_swapping_labels_changes_only_evaluation_not_model_predictions(
        self,
    ) -> None:
        """Spy on model calls and perturb labels to expose accidental leakage.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            videos = make_demo_dataset(config)
            backend = OriginalFrameSpy(config, videos)
            with patch(
                "event_caption_pilot.pipeline.build_backend",
                return_value=backend,
            ):
                with patch(
                    "event_caption_pilot.pipeline.make_demo_dataset",
                    return_value=videos,
                ):
                    original = run_experiment(config)
                flipped_videos = [
                    replace(
                        video,
                        questions=tuple(
                            replace(
                                question,
                                answer_index=1 - question.answer_index,
                            )
                            for question in video.questions
                        ),
                    )
                    for video in videos
                ]
                with patch(
                    "event_caption_pilot.pipeline.make_demo_dataset",
                    return_value=flipped_videos,
                ):
                    flipped = run_experiment(config)
            self.assertEqual(original["calibration"], flipped["calibration"])
            self.assertEqual(
                backend.caption_calls,
                sum(len(video["captions"]) for video in original["videos"]),
            )
            self.assertEqual(
                backend.qa_calls, 2 * len(original["questions"]) * 3
            )
            for original_row, flipped_row in zip(
                original["questions"], flipped["questions"], strict=True
            ):
                for condition in CONDITION_KEYS:
                    original_result = original_row["conditions"][condition]
                    flipped_result = flipped_row["conditions"][condition]
                    self.assertEqual(
                        original_result["predicted_index"],
                        flipped_result["predicted_index"],
                    )
                    self.assertEqual(
                        original_result["selected_indices"],
                        flipped_result["selected_indices"],
                    )
                    self.assertNotEqual(
                        original_result["correct"], flipped_result["correct"]
                    )

    def test_evaluation_queries_cannot_refit_development_calibration(
        self,
    ) -> None:
        """Change evaluation queries while keeping fitted statistics fixed.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            videos = make_demo_dataset(config)
            with patch(
                "event_caption_pilot.pipeline.make_demo_dataset",
                return_value=videos,
            ):
                original = run_experiment(config)
            changed_videos = [
                replace(
                    video,
                    questions=tuple(
                        replace(
                            question,
                            text=question.text + " Also consider blue.",
                        )
                        for question in video.questions
                    ),
                )
                if video.split == "eval"
                else video
                for video in videos
            ]
            with patch(
                "event_caption_pilot.pipeline.make_demo_dataset",
                return_value=changed_videos,
            ):
                changed = run_experiment(config)
            self.assertEqual(original["calibration"], changed["calibration"])
            self.assertEqual(
                set(changed["calibration"]["fit_video_ids"]),
                {video.video_id for video in videos if video.split == "dev"},
            )
