import copy
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).with_name("training_job.py")
SPEC = importlib.util.spec_from_file_location("training_job", PATH)
training_job = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(training_job)


class TrainingJobTest(unittest.TestCase):
    def config(self):
        config = copy.deepcopy(training_job.configuration(Path(__file__).parents[1] / "recipes/synthetic-bootstrap-v1.json"))
        config["dataset"]["expected_pages"] = 120
        config["dataset"]["split"] = {"algorithm": "connected-page-parent-effect-greedy-v1", "train": 0.6, "development": 0.2, "calibration": 0.2}
        return config

    def recipes(self):
        values = []
        for index in range(120):
            negative = index % 4 == 3
            values.append({"index": index, "kind": "negative" if negative else "boards", "layout": "page",
                           "condition": {"degradation": {"seed": index // 4, "variant": "blank"}},
                           "boards": [] if negative else [{"parent": f"position-{index}", "set": "original",
                           "position_kind": "teaching", "orientation": "black-bottom" if index % 2 else "white-bottom"}]})
        return values

    def test_recipe_has_one_complete_bounded_schedule(self):
        config = training_job.configuration(Path(__file__).parents[1] / "recipes/synthetic-bootstrap-v1.json")
        self.assertEqual(sum(stage["updates"] for stage in config["classifier"]["stages"]), 10000)
        self.assertEqual(sum(stage["updates"] for stage in config["detector"]["stages"]), 9000)
        self.assertEqual(config["resources"]["gpu_seconds"], 8 * 60 * 60)
        self.assertFalse(config["environment"]["cudnn_enabled"])

    def test_split_is_deterministic_and_keeps_effect_groups_together(self):
        recipes, config = self.recipes(), self.config()
        left = training_job.freeze_split(recipes, config)
        self.assertEqual(left, training_job.freeze_split(recipes, config))
        self.assertEqual(len(left["page_split"]), 120)
        for start in range(0, 120, 4):
            self.assertEqual(len({left["page_split"][str(index)] for index in range(start, start + 4)}), 1)
        for summary in left["summary"].values():
            self.assertGreater(summary["boards"], 0)
            self.assertGreater(summary["negative_pages"], 0)
            self.assertTrue(all(summary["orientations"].values()))

    def test_split_rejects_missing_orientations(self):
        recipes = self.recipes()
        for page in recipes:
            for board in page["boards"]:
                board["orientation"] = "white-bottom"
        with self.assertRaisesRegex(training_job.Invalid, "orientation"):
            training_job.freeze_split(recipes, self.config())

    def test_retries_cannot_reuse_segment_or_total_gpu_budget(self):
        resources = training_job.configuration(Path(__file__).parents[1] / "recipes/synthetic-bootstrap-v1.json")["resources"]
        state = {"gpu_seconds_charged": 5900, "attempts": [
            {"segment": "classifier", "elapsed_seconds": 5900},
        ]}
        self.assertEqual(training_job.remaining_seconds(state, "classifier", resources), 100)
        state["gpu_seconds_charged"] = resources["gpu_seconds"] - 25
        self.assertEqual(training_job.remaining_seconds(state, "classifier", resources), 25)
        state["gpu_seconds_charged"] = resources["gpu_seconds"]
        self.assertEqual(training_job.remaining_seconds(state, "classifier", resources), 0)


if __name__ == "__main__":
    unittest.main()
