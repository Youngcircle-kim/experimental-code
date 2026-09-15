"""Keep malformed observations and cross-split video leakage out of pilots."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from event_caption_pilot.config import Config
from event_caption_pilot.data import validate_dataset, validate_video
from event_caption_pilot.types import Question, Video


def make_video(video_id: str, split: str, pixel_value: int) -> Video:
    """Build a tiny original-frame pool with separate QA annotations.

    Args:
        video_id: Unique video and source identity.
        split: Development or evaluation designation.
        pixel_value: Distinguishing RGB content for leakage checks.

    Returns:
        One otherwise valid video containing four candidate frames.
    """
    return Video(
        video_id=video_id,
        split=split,
        frames=np.full((4, 2, 2, 3), pixel_value, dtype=np.uint8),
        timestamps=np.array([0.0, 1.0, 2.0, 3.0]),
        duration_seconds=4.0,
        questions=(
            Question("question", "Visible color?", ("red", "blue"), 0),
        ),
        source_id=video_id,
    )


class DataValidationTests(unittest.TestCase):
    """Verify input quality and independence before model calls."""

    def test_duplicate_source_across_splits_is_rejected(self) -> None:
        """Reject cross-split leakage even with distinct supplied crops.

        Returns:
            None.
        """
        control_video = make_video("dev_video", "dev", 10)
        treatment_video = replace(
            make_video("eval_video", "eval", 20),
            source_id=control_video.source_id,
        )
        with self.assertRaises(ValueError):
            validate_dataset(
                [control_video, treatment_video],
                replace(Config(), frame_budget=2),
            )

    def test_identical_observations_cannot_hide_behind_different_ids(
        self,
    ) -> None:
        """Reject copied candidates across splits even when IDs differ.

        Returns:
            None.
        """
        with self.assertRaises(ValueError):
            validate_dataset(
                [
                    make_video("dev_video", "dev", 10),
                    make_video("eval_video", "eval", 10),
                ],
                replace(Config(), frame_budget=2),
            )

    def test_invalid_timestamps_and_labels_fail_before_inference(self) -> None:
        """Reject repeated, nonfinite, shape-mismatched, or out-of-range data.

        Returns:
            None.
        """
        video = make_video("video", "eval", 10)
        config = replace(Config(), frame_budget=2)
        invalid_videos = (
            replace(video, timestamps=np.array([0.0, 1.0, 1.0, 3.0])),
            replace(video, timestamps=np.array([0.0, 1.0, np.nan, 3.0])),
            replace(video, timestamps=np.array([0.0, 1.0])),
            replace(
                video, questions=(replace(video.questions[0], answer_index=2),)
            ),
            replace(
                video,
                questions=(replace(video.questions[0], answer_index=True),),
            ),
            replace(
                video,
                questions=(
                    replace(
                        video.questions[0], evidence_intervals=((3.0, 2.0),)
                    ),
                ),
            ),
        )
        for invalid_video in invalid_videos:
            with self.subTest(timestamps=invalid_video.timestamps):
                with self.assertRaises(ValueError):
                    validate_video(invalid_video, config)

    def test_original_source_is_the_bootstrap_unit(self) -> None:
        """Reject splitting one source into artificial independent video IDs.

        Returns:
            None.
        """
        first_eval = make_video("eval_one", "eval", 20)
        second_eval = replace(
            make_video("eval_two", "eval", 30), source_id="eval_one"
        )
        with self.assertRaises(ValueError):
            validate_dataset(
                [make_video("dev", "dev", 10), first_eval, second_eval],
                replace(Config(), frame_budget=2),
            )

    def test_identical_eval_content_is_not_an_independent_video(self) -> None:
        """Reject duplicated evaluation content even with distinct source IDs.

        Returns:
            None.
        """
        with self.assertRaises(ValueError):
            validate_dataset(
                [
                    make_video("dev", "dev", 10),
                    make_video("eval_one", "eval", 20),
                    make_video("eval_two", "eval", 20),
                ],
                replace(Config(), frame_budget=2),
            )
