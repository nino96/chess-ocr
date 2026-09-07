import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


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
        self.assertEqual(config["resources"]["cpu_seconds"], 24000)
        self.assertFalse(config["environment"]["cudnn_enabled"])

    def test_v2_recipe_freezes_corrected_detector_only_contract(self):
        config = training_job.configuration(Path(__file__).parents[1] / "recipes/synthetic-bootstrap-v2.json")
        self.assertEqual(config["detector"]["preprocessing"], "yolox-rgb-imagenet-v2")
        self.assertEqual(config["detector"]["ema_decay"], .9998)
        self.assertEqual(config["detector"]["ema_warmup_updates"], 2000)
        self.assertEqual(config["detector"]["evaluation_steps"]["live_and_ema"], [1500, 2000, 9000])
        self.assertEqual(sum(stage["updates"] for stage in config["detector"]["stages"]), 9000)
        self.assertEqual(config["resources"]["gpu_seconds"], 6000)
        self.assertEqual(config["resources"]["preflight_gpu_seconds"], 600)
        self.assertEqual(config["resources"]["classifier_gpu_seconds"], 0)
        self.assertEqual(config["resources"]["detector_gpu_seconds"], 5400)
        self.assertEqual(config["resources"]["cpu_seconds"], 24000)

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

    def test_classifier_checkpoint_record_requires_complete_selected_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-009500.pt"
            checkpoint.write_bytes(b"checkpoint")
            development = {"exact_board_accuracy": 1.0, "occupied_macro_f1": 1.0, "nll": .01}
            final = {"exact_board_accuracy": .9, "occupied_macro_f1": .9, "nll": .02}
            training_job.write_json(root / "curves.json", [
                {"global_step": 9500, "development": development},
                {"global_step": 10000, "development": final},
            ])
            training_job.write_json(root / "progress.json", {"global_step": 10000})
            record = training_job.classifier_checkpoint_record(checkpoint, self.config())
            self.assertEqual(record["selected_global_step"], 9500)
            self.assertEqual(record["development"], development)
            training_job.write_json(root / "progress.json", {"global_step": 3500})
            with self.assertRaisesRegex(training_job.Invalid, "incomplete"):
                training_job.classifier_checkpoint_record(checkpoint, self.config())

    def test_container_uses_host_identity_and_dedicated_output_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "overlay"; overlay.mkdir()
            frozen = {"run_id": "a" * 64, "repository_root": str(root), "dataset_root": str(root),
                      "native_root": str(root), "dependency_overlay_root": str(overlay),
                      "container_user": {"uid": 123, "gid": 456}, "recipe": self.config()}
            command = training_job.container_command(root, frozen, "preflight")
            self.assertIn("123:456", command)
            self.assertIn(f"{root}:/output:rw", command)
            self.assertNotIn(f"{root}:/run:rw", command)
            self.assertEqual(command[command.index("--entrypoint") + 1], "python")
            validation = training_job.container_command(root, frozen, "validate")
            self.assertNotIn("--gpus", validation)
            self.assertIn("validate", validation)

    def test_detector_only_container_mounts_checkpoint_without_gpu_for_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "overlay"; overlay.mkdir()
            checkpoint = root / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
            frozen = {"run_id": "e" * 64, "repository_root": str(root), "dataset_root": str(root),
                      "native_root": str(root), "dependency_overlay_root": str(overlay),
                      "container_user": {"uid": 123, "gid": 456}, "recipe": self.config(),
                      "mode": "detector-only", "classifier_checkpoint":
                      {"path": str(checkpoint), "sha256": training_job.sha256(checkpoint)}}
            command = training_job.container_command(root, frozen, "classifier-export")
            self.assertNotIn("--gpus", command)
            self.assertIn("/classifier-checkpoint.pt", command)
            self.assertIn("classifier-export", command)

    def test_detector_export_retry_is_cpu_only_and_uses_persisted_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "overlay"; overlay.mkdir()
            detector = root / "detector"; detector.mkdir()
            model = detector / "selected.onnx"; model.write_bytes(b"onnx")
            training_job.write_json(detector / "progress.json", {"state": "export_failed", "selected_global_step": 9000,
                "final": {"ap50_95": .5}, "calibration": {"threshold": .1}})
            training_job.write_json(detector / "selected.manifest.json", {"sha256": training_job.sha256(model)})
            frozen = {"run_id": "f" * 64, "repository_root": str(root), "dataset_root": str(root),
                      "native_root": str(root), "dependency_overlay_root": str(overlay),
                      "container_user": {"uid": 123, "gid": 456}, "recipe": self.config()}
            training_job.write_json(root / "state.json", {"state": "export_failed", "lifecycle_state": "export_failed",
                                                           "pid": None, "process_start": None, "attempts": []})
            with mock.patch.object(training_job, "verify_frozen", return_value=(frozen, self.config())), \
                 mock.patch.object(training_job.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as execute:
                result = training_job.export(root, "detector")
            self.assertEqual(result["state"], "complete")
            command = execute.call_args.args[0]
            self.assertNotIn("--gpus", command)
            self.assertIn("detector-export", command)
            self.assertEqual(training_job.read_json(root / "state.json")["lifecycle_state"], "complete")

    def test_detector_export_retry_rejects_missing_durable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "detector").mkdir()
            training_job.write_json(root / "detector" / "progress.json", {"state": "export_failed", "selected_global_step": 9000})
            with self.assertRaisesRegex(training_job.Invalid, "final evaluation"):
                training_job.detector_export_evidence(root)

    def test_start_refuses_to_repeat_work_after_export_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_job.write_json(root / "state.json", {
                "state": "export_failed", "pid": None, "process_start": None})
            with mock.patch.object(training_job, "verify_frozen", return_value=({}, self.config())), \
                 mock.patch.object(training_job, "validate_before_gpu") as validate:
                with self.assertRaisesRegex(training_job.Invalid, "export-only retry"):
                    training_job.start(root)
            validate.assert_not_called()

    def test_status_defaults_to_current_attempt_and_reports_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_job.write_json(root / "state.json", {
                "state": "running", "stage": "classifier", "pid": None,
                "process_start": None, "gpu_seconds_charged": 125,
                "attempts": [{"segment": "preflight", "elapsed_seconds": 25},
                             {"segment": "classifier", "elapsed_seconds": 100}],
            })
            training_job.write_json(root / "frozen.json", {"recipe": {"resources": {
                "gpu_seconds": 1000, "preflight_gpu_seconds": 100,
                "classifier_gpu_seconds": 300, "detector_gpu_seconds": 600}}})
            result = training_job.status(root)
            self.assertNotIn("attempts", result)
            self.assertEqual(result["current_attempt"]["segment"], "classifier")
            self.assertEqual(result["budget"]["capacity_seconds"], 1000)
            self.assertEqual(result["budget"]["consumed_seconds"], 125)
            self.assertEqual(result["budget"]["by_segment"]["classifier"]["remaining_seconds"], 200)

    def test_status_includes_live_unfinalized_gpu_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_job.write_json(root / "state.json", {
                "state": "running", "stage": "detector", "pid": 123, "process_start": "token",
                "gpu_seconds_charged": 25, "attempts": [],
                "operation": {"name": "detector", "resource": "GPU", "started_at": 900},
            })
            training_job.write_json(root / "frozen.json", {"recipe": {"resources": {
                "gpu_seconds": 1000, "preflight_gpu_seconds": 100,
                "classifier_gpu_seconds": 300, "detector_gpu_seconds": 600}}})
            with mock.patch.object(training_job, "process_start", return_value="token"), \
                 mock.patch.object(training_job.time, "time", return_value=1000):
                result = training_job.status(root)
            self.assertEqual(result["budget"]["charged_seconds"], 25)
            self.assertEqual(result["budget"]["active_unfinalized_seconds"], 100)
            self.assertEqual(result["budget"]["consumed_seconds"], 125)

    def test_status_hides_inherited_attempts_and_history_deduplicates_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_job.write_json(root / "state.json", {"state": "ready", "pid": None, "process_start": None,
                "gpu_seconds_charged": 150, "attempts": [
                    {"segment": "detector", "elapsed_seconds": 100, "inherited": True, "ledger_id": "old:0"},
                    {"segment": "detector", "elapsed_seconds": 100, "inherited": True, "ledger_id": "old:0"},
                    {"segment": "preflight", "elapsed_seconds": 50, "ledger_id": "new:0"},
                ]})
            training_job.write_json(root / "frozen.json", {"mode": "detector-only", "classifier_checkpoint": {
                "sha256": "checkpoint", "selected_global_step": 10000, "development": {"nll": .1}}, "recipe": {"resources": {
                "gpu_seconds": 1000, "preflight_gpu_seconds": 100, "classifier_gpu_seconds": 300,
                "detector_gpu_seconds": 600, "diagnosis_gpu_seconds": 0}}})
            training_job.write_json(root / "classifier" / "progress.json", {"state": "complete"})
            current = training_job.status(root)
            self.assertEqual(current["budget"]["charged_seconds"], 50)
            self.assertNotIn("classifier", current["budget"]["by_segment"])
            self.assertEqual(current["classifier"]["state"], "reused")
            history = training_job.status(root, history=True)
            self.assertEqual(len(history["attempts"]), 2)
            self.assertEqual(history["history_budget"]["charged_seconds"], 150)

    def test_failed_cpu_validation_never_requests_gpu_or_writes_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "overlay"; overlay.mkdir()
            frozen = {"run_id": "b" * 64, "repository_root": str(root), "dataset_root": str(root),
                      "native_root": str(root), "dependency_overlay_root": str(overlay),
                      "container_user": {"uid": 123, "gid": 456}, "recipe": self.config()}
            with mock.patch.object(training_job.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as execute:
                with self.assertRaisesRegex(training_job.Invalid, "no GPU charged"):
                    training_job.validate_before_gpu(root, frozen)
            self.assertNotIn("--gpus", execute.call_args.args[0])
            self.assertFalse((root / "validation.complete.json").exists())

    def test_cpu_validation_timeout_stops_its_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "overlay"; overlay.mkdir()
            frozen = {"run_id": "c" * 64, "repository_root": str(root), "dataset_root": str(root),
                      "native_root": str(root), "dependency_overlay_root": str(overlay),
                      "container_user": {"uid": 123, "gid": 456}, "recipe": self.config()}
            timeout = training_job.subprocess.TimeoutExpired(["docker", "run"], 300)
            with mock.patch.object(training_job.subprocess, "run", side_effect=timeout), \
                 mock.patch.object(training_job, "stop_container") as stop:
                with self.assertRaisesRegex(training_job.Invalid, "container stopped"):
                    training_job.validate_before_gpu(root, frozen)
            stop.assert_called_once_with(frozen, "validate")

    def test_validation_failure_blocks_supervisor_and_persists_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_job.write_json(root / "state.json", {"state": "ready", "pid": None})
            frozen = {"run_id": "d" * 64}
            failure = training_job.Invalid("CPU prelaunch validation failed")
            with mock.patch.object(training_job, "verify_frozen", return_value=(frozen, self.config())), \
                 mock.patch.object(training_job, "validate_before_gpu", side_effect=failure), \
                 mock.patch.object(training_job.subprocess, "Popen") as supervisor:
                with self.assertRaisesRegex(training_job.Invalid, "validation failed"):
                    training_job.start(root)
            supervisor.assert_not_called()
            state = training_job.read_json(root / "state.json")
            self.assertEqual(state["state"], "validation-failed")


if __name__ == "__main__":
    unittest.main()
