"""Test the new handoff with mocks only: no BERT, CUDA, Slurm or downloads."""
import copy
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "slurm/smoke_then_baseline_seed42.sbatch"
HANDOFF = SCRIPT.read_text().split('"$PYTHON" -u - <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]


class HandoffTests(unittest.TestCase):
    def invoke(self, change=None, artifact_failure=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            smoke_dir = root / "outputs/smoke/seed42/job123"
            smoke_dir.mkdir(parents=True)
            (smoke_dir / "pip-freeze.txt").write_text("torch==2.6.0+cu118\n")
            full_config = {"seed": 42, "epochs": 4, "per_device_train_batch_size": 16}
            smoke_config = {**full_config, "mode": "smoke", "epochs": 1}
            meta = {"slurm_job_id": "123", "source_sha256": {"src/train.py": "same"},
                    "assets_manifest_sha256": "asset"}
            state = {"meta": meta, "smoke": smoke_config, "freeze": "torch==2.6.0+cu118\n"}
            if change:
                change(state)
            common = types.ModuleType("src.common")
            common.ROOT = root
            common.load_config = lambda: copy.deepcopy(full_config)
            common.sha256 = lambda path: "asset"
            common.source_hashes = lambda: {"src/train.py": "same"}
            verifier = types.ModuleType("scripts.verify_run")
            verifier.verify_artifacts = Mock(return_value=(meta, smoke_config))
            if artifact_failure:
                verifier.verify_artifacts.side_effect = RuntimeError("Missing checkpoint")
            trainer = types.ModuleType("src.train")
            trainer.run = Mock()
            modules = {"src.common": common, "scripts.verify_run": verifier, "src.train": trainer}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), \
                    patch.object(subprocess, "check_output", return_value=state["freeze"]):
                if change or artifact_failure:
                    with self.assertRaises(RuntimeError):
                        exec(compile(HANDOFF, str(SCRIPT), "exec"), {})
                    trainer.run.assert_not_called()
                else:
                    exec(compile(HANDOFF, str(SCRIPT), "exec"), {})
                    trainer.run.assert_called_once_with({**full_config, "mode": "full"}, root / "outputs/bert_baseline/seed42")

    def test_success_starts_full_four_epochs(self):
        self.invoke()

    def test_missing_artifacts_block_full(self):
        self.invoke(artifact_failure=True)

    def test_other_job_smoke_is_rejected(self):
        self.invoke(lambda s: s["meta"].update(slurm_job_id="122"))

    def test_changed_code_blocks_full(self):
        self.invoke(lambda s: s["meta"].update(source_sha256={"src/train.py": "changed"}))

    def test_changed_assets_block_full(self):
        self.invoke(lambda s: s["meta"].update(assets_manifest_sha256="changed"))

    def test_changed_packages_block_full(self):
        self.invoke(lambda s: s.update(freeze="torch==different\n"))

    def test_changed_batch_blocks_full(self):
        self.invoke(lambda s: s["smoke"].update(per_device_train_batch_size=8))


if __name__ == "__main__":
    unittest.main()
