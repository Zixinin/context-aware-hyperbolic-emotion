"""Disk persistence + mocked HTTP Range tests; no real downloads or GPU."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("resume_download", Path(__file__).parents[1] / "scripts/resume_download.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BODY = b"abcdef" * 200
SHA = hashlib.sha256(BODY).hexdigest()


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body))}


class Interrupted(Response):
    def read(self, size=-1):
        if self.tell() == 0:
            return super().read(300)
        raise KeyboardInterrupt("simulate process stopped after partial download")


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "asset.bin"

    def interrupt(self):
        with patch.object(module.urllib.request, "urlopen", return_value=Interrupted(BODY)):
            with self.assertRaises(KeyboardInterrupt):
                module.download("https://example.org/file", self.output, SHA)
        self.assertEqual(self.output.with_name("asset.bin.part").stat().st_size, 300)
        self.assertFalse(self.output.exists())

    def test_resume_on_second_invocation_and_skip_completed(self):
        self.interrupt()
        def serve(request, **kwargs):
            self.assertEqual(request.get_header("Range"), "bytes=300-")
            return Response(BODY[300:], 206, {"Content-Range": f"bytes 300-{len(BODY)-1}/{len(BODY)}"})
        with patch.object(module.urllib.request, "urlopen", side_effect=serve):
            module.download("https://example.org/file", self.output, SHA)
        self.assertEqual(self.output.read_bytes(), BODY)
        with patch.object(module.urllib.request, "urlopen") as mocked:
            module.download("https://example.org/file", self.output, SHA)
            mocked.assert_not_called()

    def test_server_ignoring_range_restarts_only_this_file(self):
        self.interrupt()
        with patch.object(module.urllib.request, "urlopen", return_value=Response(BODY)):
            module.download("https://example.org/file", self.output, SHA)
        self.assertEqual(self.output.read_bytes(), BODY)

    def test_wrong_content_range_not_appended(self):
        self.interrupt()
        with patch.object(module.urllib.request, "urlopen", return_value=Response(BODY, 206, {"Content-Range": "bytes 0-1199/1200"})):
            with self.assertRaises(ValueError):
                module.download("https://example.org/file", self.output, SHA)
        self.assertEqual(self.output.with_name("asset.bin.part").stat().st_size, 300)

    def test_wrong_hash_not_promoted(self):
        with patch.object(module.urllib.request, "urlopen", return_value=Response(b"wrong")):
            with self.assertRaises(ValueError):
                module.download("https://example.org/file", self.output, SHA)
        self.assertFalse(self.output.exists())

    def test_changed_asset_does_not_reuse_partial(self):
        self.interrupt()
        with self.assertRaises(ValueError):
            module.download("https://example.org/other", self.output, "0" * 64)


if __name__ == "__main__":
    unittest.main()
