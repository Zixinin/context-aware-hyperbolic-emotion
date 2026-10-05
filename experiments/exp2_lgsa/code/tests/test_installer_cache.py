import importlib.util
import ast
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('installer', Path(__file__).parents[1] / 'environment/install_all_cpu.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class CacheTests(unittest.TestCase):
    def test_resolver_output_with_local_torch(self):
        # Execute the installer's actual report loop against uv-style output.
        tree = ast.parse(Path(installer.__file__).read_text())
        loop = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.For)
                    and ast.unparse(node.iter) == 'report_path.read_text().splitlines()')
        code = compile(ast.Module(body=[loop], type_ignores=[]), '<report-loop>', 'exec')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wheel = base / 'torch-2.5.1+cu118-cp310-cp310-linux_x86_64.whl'
            report = base / 'versions.txt'
            for torch_line in (str(wheel), f'torch @ {wheel.as_uri()}'):
                with self.subTest(torch_line=torch_line):
                    report.write_text(f'# compiled dependencies\ntokenizers==0.19.1\n{torch_line}\ntqdm==4.66.5\n')
                    pins = []
                    namespace = dict(vars(installer), report_path=report,
                                     torch_wheel=wheel, metadata_dir=base,
                                     wheels=base, pins=pins)
                    with patch.object(installer, 'package_file', return_value={
                        'url': 'https://example.org/wheel', 'filename': 'test.whl',
                        'digests': {'sha256': '0' * 64}
                    }) as metadata, patch.object(installer, 'fetch_wheel') as fetch:
                        namespace.update(package_file=metadata, fetch_wheel=fetch)
                        exec(code, namespace)
                        self.assertEqual(metadata.call_count, 2)
                        self.assertEqual(fetch.call_count, 2)
                    self.assertEqual([pin.split()[0] for pin in pins],
                                     ['tokenizers==0.19.1', 'tqdm==4.66.5'])
            report.write_text(str(base / 'unexpected.whl') + '\n')
            with self.assertRaisesRegex(AssertionError, 'Unexpected local dependency'):
                exec(code, dict(vars(installer), report_path=report, torch_wheel=wheel))

    def test_process_interruption_then_resume_and_skip(self):
        data = b'complete-wheel-content'
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'test.whl'
            part = target.with_name(target.name + '.part')
            def interrupted(args):
                part.write_bytes(data[:8])
                raise KeyboardInterrupt()
            with patch.object(installer, 'run', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    installer.fetch_wheel('https://example.org/test.whl', digest, target)
            self.assertEqual(part.read_bytes(), data[:8])
            def resumed(args):
                self.assertIn('--continue-at', args)
                with part.open('ab') as stream:
                    stream.write(data[8:])
            with patch.object(installer, 'run', side_effect=resumed):
                installer.fetch_wheel('https://example.org/test.whl', digest, target)
            self.assertEqual(target.read_bytes(), data)
            with patch.object(installer, 'run') as mock:
                installer.fetch_wheel('https://example.org/test.whl', digest, target)
                mock.assert_not_called()

    def test_hash_mismatch_never_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'test.whl'
            part = target.with_name(target.name + '.part')
            with patch.object(installer, 'run', side_effect=lambda args: part.write_bytes(b'wrong')):
                with self.assertRaises(ValueError):
                    installer.fetch_wheel('https://example.org/test.whl', '0' * 64, target)
            self.assertFalse(target.exists())
            self.assertTrue(part.exists())

    def test_changed_version_preserves_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'test.whl'
            target.with_name(target.name + '.part').write_bytes(b'partial')
            target.with_name(target.name + '.part.sha256').write_text('1' * 64)
            with self.assertRaises(ValueError):
                installer.fetch_wheel('https://example.org/test.whl', '0' * 64, target)


if __name__ == '__main__':
    unittest.main()
