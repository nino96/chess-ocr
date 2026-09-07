#!/usr/bin/env python3
"""Adversarial regression tests for issue #3 detector reporting."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


SPEC = importlib.util.spec_from_file_location("training", Path(__file__).with_name("training.py"))
assert SPEC and SPEC.loader
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


class Pages:
    def __init__(self, metadata: list[dict]):
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def get(self, index: int):
        return torch.zeros((3, 416, 416)), torch.zeros((8, 5)), self.metadata[index]


class FixedDetector(nn.Module):
    def __init__(self, output: list[list[list[float]]]):
        super().__init__()
        self.output = torch.tensor(output, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output[:len(inputs)].to(inputs.device)


def page(boxes: list[list[float]], *, negative: bool = False) -> dict:
    return {"boxes": boxes, "negative": negative, "effect": "clean", "layout": "single"}


class DetectorMetricTest(unittest.TestCase):
    def test_score_ordered_matching_is_one_to_one(self) -> None:
        predicted = torch.tensor([[0, 0, 10, 10, .9], [0, 0, 10, 10, .8]])
        truth = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual(training.score_ordered_matches(predicted, truth, .5),
                         [(0, 0, 1.0), (1, 1, 1.0)])
        self.assertEqual(training.score_ordered_matches(predicted[:1], truth, .5), [(0, 0, 1.0)])

    def test_confidence_tie_preserves_decoder_order(self) -> None:
        predicted = torch.tensor([[0, 0, 10, 10, .9], [8, 0, 18, 10, .9]])
        truth = torch.tensor([[0, 0, 10, 10], [8, 0, 18, 10]])
        self.assertEqual(training.score_ordered_matches(predicted, truth, .5),
                         [(0, 0, 1.0), (1, 1, 1.0)])

    def test_missed_targets_remain_in_metric_denominators(self) -> None:
        dataset = Pages([page([[40, 40, 60, 60], [40, 40, 60, 60]])])
        metrics = training.evaluate_detector(FixedDetector([[[50, 50, 20, 20, 1, 1]]]),
                                             dataset, torch.device("cpu"), .5)
        self.assertEqual(metrics["recall"]["0.5"], .5)
        self.assertEqual(metrics["mean_iou_at_0_5_including_misses"], .5)
        self.assertEqual(metrics["strata"]["layout"]["single"]["targets"], 2)

    def test_empty_predictions_and_negative_false_positives_are_counted(self) -> None:
        positive = Pages([page([[40, 40, 60, 60]])])
        metrics = training.evaluate_detector(FixedDetector([[[50, 50, 20, 20, 0, 1]]]),
                                             positive, torch.device("cpu"), .5)
        self.assertEqual(metrics["recall"]["0.5"], 0.0)
        self.assertEqual(metrics["mean_iou_at_0_5_including_misses"], 0.0)

        negative = Pages([page([], negative=True)])
        metrics = training.evaluate_detector(FixedDetector([[[50, 50, 20, 20, 1, 1]]]),
                                             negative, torch.device("cpu"), .5)
        self.assertEqual(metrics["false_detections_on_negative_pages_at_0_01"], 1)
        self.assertEqual(metrics["precision"]["0.5"], 0.0)

    def test_calibration_cannot_count_one_detection_twice(self) -> None:
        dataset = Pages([page([[40, 40, 60, 60], [40, 40, 60, 60]])])
        calibration = training.calibrate_detector(FixedDetector([[[50, 50, 20, 20, 1, .8]]]),
                                                    dataset, torch.device("cpu"), .5)
        self.assertEqual(calibration["recall"], .5)


if __name__ == "__main__":
    unittest.main()
