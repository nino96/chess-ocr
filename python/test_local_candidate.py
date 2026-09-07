"""Pure validation tests for the optional local candidate adapter."""
import unittest

from python.local_candidate import Invalid, LocalCandidate


def manifest():
    return {
        "schema": "chess-ocr-candidate-bundle/1",
        "name": "test",
        "version": "1",
        "qualification": "synthetic-development-only",
        "classifier": {"sha256": "a" * 64, "bytes": 1024, "input": "tiles",
                       "output": "logits",
                       "labels": ["empty", "P", "N", "B", "R", "Q", "K",
                                  "p", "n", "b", "r", "q", "k"]},
        "detector": {"sha256": "b" * 64, "bytes": 1024, "input": "images",
                     "output": "predictions", "scoreThreshold": .3, "nmsIou": .65},
    }


class LocalCandidateTests(unittest.TestCase):
    def test_manifest_is_closed_and_synthetic_only(self):
        LocalCandidate._validate_manifest(manifest())
        with self.assertRaisesRegex(Invalid, "unsupported"):
            LocalCandidate._validate_manifest({**manifest(), "qualification": "production"})
        with self.assertRaisesRegex(Invalid, "invalid candidate manifest"):
            LocalCandidate._validate_manifest({**manifest(), "extra": True})

    def test_label_order_threshold_and_iou_are_fixed(self):
        value = manifest()
        value["classifier"]["labels"] = list(reversed(value["classifier"]["labels"]))
        with self.assertRaisesRegex(Invalid, "classifier labels"):
            LocalCandidate._validate_manifest(value)
        value = manifest()
        value["detector"]["scoreThreshold"] = 0
        with self.assertRaisesRegex(Invalid, "score threshold"):
            LocalCandidate._validate_manifest(value)
        self.assertEqual(LocalCandidate._iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(LocalCandidate._iou([0, 0, 10, 10], [10, 10, 20, 20]), 0)


if __name__ == "__main__":
    unittest.main()
