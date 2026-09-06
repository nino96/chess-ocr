import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("native_runtime.py")
SPEC = importlib.util.spec_from_file_location("native_runtime", MODULE_PATH)
native_runtime = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(native_runtime)


class FakeResponse:
    def __init__(self, url, payload):
        self.url = url
        self.headers = {"Content-Length": str(len(payload))}
        self.payload = payload

    def read(self, size):
        if not self.payload:
            return b""
        result, self.payload = self.payload[:size], self.payload[size:]
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class NativeRuntimeTest(unittest.TestCase):
    def test_rejects_unexpected_redirect_without_writing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.safetensors"
            response = FakeResponse("https://example.invalid/model", b"payload")
            with patch.object(native_runtime.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "unexpected final download host"):
                    native_runtime.checked_download("https://huggingface.co/x", destination, "huggingface.co", hashlib.sha256(b"payload").hexdigest())
            self.assertFalse(destination.exists())

    def test_enforces_total_cache_download_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            work = Path(directory) / "work"
            work.mkdir()
            (work / "download-budget.json").write_text('{"downloaded_bytes": 8}\n')
            destination = cache / "next"
            response = FakeResponse("https://huggingface.co/model", b"abc")
            response.headers = {}
            with patch.object(native_runtime, "CACHE", cache), patch.object(native_runtime, "WORK", work), patch.object(native_runtime, "MAX_DOWNLOAD_BYTES", 10), patch.object(native_runtime.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "total exceeded"):
                    native_runtime.checked_download("https://huggingface.co/x", destination, "huggingface.co", hashlib.sha256(b"abc").hexdigest())
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".partial").exists())
            self.assertEqual(json.loads((work / "download-budget.json").read_text())["downloaded_bytes"], 11)

    def test_rejects_bad_hash_and_symlink_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            work = Path(directory) / "work"
            destination = cache / "model.safetensors"
            response = FakeResponse("https://huggingface.co/model", b"payload")
            with patch.object(native_runtime, "CACHE", cache), patch.object(native_runtime, "WORK", work), patch.object(native_runtime.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    native_runtime.checked_download("https://huggingface.co/x", destination, "huggingface.co", hashlib.sha256(b"other").hexdigest())
            self.assertFalse(destination.exists())
            target = Path(directory) / "outside"
            target.write_bytes(b"outside")
            destination.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlinked"):
                native_runtime.checked_download("https://huggingface.co/x", destination, "huggingface.co", hashlib.sha256(b"payload").hexdigest())

    def test_generator_records_uint8_and_float_preprocessing_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(native_runtime, "ROOT", root), patch.object(native_runtime, "WORK", root / "work"), patch.object(native_runtime, "ARTIFACTS", root / "artifacts"):
                with redirect_stdout(StringIO()):
                    native_runtime.prepare_inputs()
            manifest = json.loads((root / "work/parity-input-manifest.json").read_text())
            self.assertEqual(manifest["input"]["sha256"], "41738a6eb16d0273bca34ed13ecd8a06b2bc7ea093234d3e76047654e7d2c0a3")
            self.assertEqual((root / manifest["mobilenet"]["cropped_rgb_path"]).stat().st_size, 224 * 224 * 3)
            self.assertEqual((root / manifest["mobilenet"]["tensor_path"]).stat().st_size, 1 * 3 * 224 * 224 * 4)
            self.assertEqual((root / manifest["yolox"]["tensor_path"]).stat().st_size, 1 * 3 * 416 * 416 * 4)


if __name__ == "__main__":
    unittest.main()
