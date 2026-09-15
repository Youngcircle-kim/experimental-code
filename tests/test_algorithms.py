"""Numerical and experimental-design invariants for the minimal pilot."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from event_caption_pilot.algorithms import (
    ScoreCalibrator,
    allocate_frames,
    clustered_comparison,
    event_features,
    evidence_metrics,
    normalize_rows,
    sample_indices,
    segment,
)
from event_caption_pilot.config import Config
from event_caption_pilot.types import Event


class SegmentationTests(unittest.TestCase):
    """Check partitions, constant observations, and actual boundary changes."""

    def test_uniform_partition_matches_requested_detector_count(self) -> None:
        """Cover each original index once with the requested count.

        Returns:
            None.
        """
        features = np.tile([1.0, 0.0], (19, 1))
        config = replace(Config(), uniform_segments=3, min_segment_frames=2)
        events = segment(features, config, "D0", segment_count=4)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].start, 0)
        self.assertEqual(events[-1].stop, len(features))
        covered = [
            index
            for event in events
            for index in range(event.start, event.stop)
        ]
        self.assertEqual(covered, list(range(len(features))))
        self.assertLessEqual(
            max(event.stop - event.start for event in events)
            - min(event.stop - event.start for event in events),
            1,
        )

    def test_constant_observations_have_no_false_boundary(self) -> None:
        """Ensure both feature-difference detectors retain a flat event.

        Returns:
            None.
        """
        config = replace(Config(), boundary_threshold=0.0)
        for direction in ([1.0, 0.0], [1.0, 1.0], [1.0, 2.0, 3.0]):
            features = np.tile(direction, (12, 1))
            for detector in ("D1", "D2"):
                with self.subTest(detector=detector, direction=direction):
                    self.assertEqual(
                        segment(features, config, detector), [Event(0, 12)]
                    )

    def test_change_threshold_and_minimum_segment_length(self) -> None:
        """Find a supported change and suppress weak or too-short changes.

        Returns:
            None.
        """
        features = np.vstack(
            (np.tile([1.0, 0.0], (6, 1)), np.tile([0.0, 1.0], (6, 1)))
        )
        config = replace(
            Config(),
            boundary_threshold=0.5,
            window_size=2,
            min_segment_frames=3,
        )
        for detector in ("D1", "D2"):
            with self.subTest(detector=detector):
                self.assertEqual(
                    segment(features, config, detector),
                    [Event(0, 6), Event(6, 12)],
                )
                self.assertEqual(
                    segment(
                        features,
                        replace(config, boundary_threshold=1.1),
                        detector,
                    ),
                    [Event(0, 12)],
                )
        short_change = np.vstack(([[1.0, 0.0]], np.tile([0.0, 1.0], (11, 1))))
        self.assertEqual(segment(short_change, config, "D1"), [Event(0, 12)])

    def test_nonfinite_and_empty_model_features_fail(self) -> None:
        """Reject invalid features before cosine scores or segmentation.

        Returns:
            None.
        """
        invalid_features = (
            np.empty((0, 2)),
            np.array([[1.0, float("nan")]]),
            np.array([[float("inf"), 0.0]]),
        )
        for features in invalid_features:
            with self.subTest(shape=features.shape):
                with self.assertRaises(ValueError):
                    segment(features, Config(), "D1")

    def test_event_features_preserve_unit_rows(self) -> None:
        """Pool observations by event without changing row count or norms.

        Returns:
            None.
        """
        features = np.array([[2.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        pooled = event_features(features, [Event(0, 2), Event(2, 3)], 1e-8)
        self.assertEqual(pooled.shape, (2, 2))
        np.testing.assert_allclose(pooled, np.eye(2))
        with self.assertRaises(ValueError):
            normalize_rows(np.zeros((2, 3)), 1e-8)


class AllocationTests(unittest.TestCase):
    """Ensure every comparison obeys the same unique original-frame budget."""

    def test_allocation_exact_budget_with_capacity_limits(self) -> None:
        """Respect capacities even when the highest-relevance event is full.

        Returns:
            None.
        """
        capacities = np.array([1, 2, 8, 4], dtype=np.int64)
        scores = np.array([100.0, 2.0, 1.0, -1.0])
        for budget in (1, 4, 9, 15):
            with self.subTest(budget=budget):
                allocation = allocate_frames(scores, capacities, budget, 1.0)
                self.assertEqual(allocation.shape, capacities.shape)
                self.assertEqual(int(allocation.sum()), budget)
                self.assertTrue(np.issubdtype(allocation.dtype, np.integer))
                self.assertTrue(np.all(allocation >= 0))
                self.assertTrue(np.all(allocation <= capacities))

    def test_more_events_than_budget_still_selects_unique_frames(self) -> None:
        """Allow zero-allocation events when a per-event floor is impossible.

        Returns:
            None.
        """
        capacities = np.full(10, 3, dtype=np.int64)
        allocation = allocate_frames(
            np.arange(10, dtype=float), capacities, 4, 1.0
        )
        selected = np.concatenate(
            [
                sample_indices(index * 3, (index + 1) * 3, int(count))
                for index, count in enumerate(allocation)
            ]
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(np.unique(selected)), 4)
        self.assertTrue(np.all(selected[:-1] < selected[1:]))

    def test_impossible_or_nonfinite_allocations_fail(self) -> None:
        """Reject impossible budgets and contaminated relevance scores.

        Returns:
            None.
        """
        with self.assertRaises(ValueError):
            allocate_frames(np.array([1.0, 2.0]), np.array([1, 1]), 3, 1.0)
        with self.assertRaises(ValueError):
            allocate_frames(
                np.array([float("nan"), 2.0]), np.array([2, 2]), 2, 1.0
            )
        with self.assertRaises(ValueError):
            sample_indices(3, 5, 3)


class StatisticalTests(unittest.TestCase):
    """Verify development calibration and paired video-cluster analysis."""

    def test_calibration_is_fitted_once_and_finite_for_constant_scores(
        self,
    ) -> None:
        """Ensure evaluation transforms do not refit development statistics.

        Returns:
            None.
        """
        calibrator = ScoreCalibrator.fit(
            np.array([0.0, 1.0, 2.0]), np.array([10.0, 10.0, 10.0]), 1e-8
        )
        initial_state = (
            calibrator.visual_mean,
            calibrator.visual_std,
            calibrator.text_mean,
            calibrator.text_std,
        )
        first_visual, first_text = calibrator.transform(
            np.array([1.0]), np.array([10.0])
        )
        calibrator.transform(np.array([10000.0]), np.array([-10000.0]))
        repeated_visual, repeated_text = calibrator.transform(
            np.array([1.0]), np.array([10.0])
        )
        np.testing.assert_array_equal(first_visual, repeated_visual)
        np.testing.assert_array_equal(first_text, repeated_text)
        self.assertTrue(np.isfinite(first_text).all())
        self.assertEqual(
            initial_state,
            (
                calibrator.visual_mean,
                calibrator.visual_std,
                calibrator.text_mean,
                calibrator.text_std,
            ),
        )

    def test_evidence_metrics_distinguish_partial_and_all_interval_hits(
        self,
    ) -> None:
        """Count intervals containing selected timestamps, including endpoints.

        Returns:
            None.
        """
        intervals = ((0.0, 1.0), (3.0, 4.0))
        partial = evidence_metrics(np.array([1.0, 2.0]), intervals)
        self.assertEqual(partial["interval_hit"], 0.5)
        self.assertFalse(partial["all_interval_hit"])
        complete = evidence_metrics(np.array([0.0, 4.0]), intervals)
        self.assertEqual(complete["interval_hit"], 1.0)
        self.assertTrue(complete["all_interval_hit"])
        missing = evidence_metrics(np.array([1.0]), ())
        self.assertIsNone(missing["interval_hit"])
        self.assertIsNone(missing["all_interval_hit"])

    def test_bootstrap_keeps_questions_from_each_video_together(self) -> None:
        """Expose pseudoreplication with perfectly dependent within-video rows.

        Returns:
            None.
        """
        rows = [
            {
                "video_id": f"video_{video_index}",
                "question_id": f"q_{question_index}",
                "control_v_correct": bool(video_index),
                "treatment_t_correct": bool(video_index),
                "treatment_vt_correct": not bool(video_index),
            }
            for video_index in range(2)
            for question_index in range(20)
        ]
        config = replace(Config(), bootstrap_samples=1000, seed=19)
        metrics = clustered_comparison(rows, config)
        self.assertEqual(metrics, clustered_comparison(rows, config))
        self.assertEqual(metrics["n_questions"], 40)
        self.assertEqual(metrics["n_videos"], 2)
        self.assertEqual(metrics["resampling_unit"], "video")
        paired = metrics["paired_comparisons"]["treatment_vt_minus_control_v"]
        self.assertEqual(paired["mean_difference"], 0.0)
        np.testing.assert_array_equal(
            paired["confidence_interval"], [-1.0, 1.0]
        )
