"""Synthetic scientific-protocol checks; no project data or network accessed."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import run_exp4a as ex

ex.load_science()
np = ex.np


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9)
        self.x = self.rng.binomial(3, .1, size=(200, 27)) / 3
        self.r, self.d = ex.correlations(self.x)
        self.z = ex.linkage_from(self.d)

    def test_official_distance_equivalence(self):
        np.testing.assert_allclose(ex.distance.pdist(self.x.T, "correlation"), ex.distance.squareform(self.d), atol=1e-12)

    def test_zero_variance_rejected(self):
        self.x[:, 0] = 0
        with self.assertRaisesRegex(RuntimeError, "Zero-variance"):
            ex.correlations(self.x)

    def test_exact_k_and_tree_leaf_integrity(self):
        for k in ex.KS:
            self.assertEqual(len(set(ex.cut(self.z, k))), k)
        obj = ex.tree(self.z, "raw")
        ex.validate_tree(obj)
        obj["root"]["children"][1]["children"][0]["node_id"] = "anger"
        with self.assertRaises(RuntimeError):
            ex.validate_tree(obj)

    def test_sensitivity_does_not_overwrite_secondary(self):
        with tempfile.TemporaryDirectory(prefix="exp4a_synthetic_") as directory:
            out = Path(directory)
            ex.save_analysis(out, "secondary_simplified", self.r, self.d, self.z)
            before = ex.sha(out / "secondary_simplified_distance.npy")
            ex.save_analysis(out, "sensitivity_official_row_filter", self.r, self.d / 2, self.z)
            self.assertEqual(before, ex.sha(out / "secondary_simplified_distance.npy"))

    def test_notification_failure_nonfatal(self):
        with patch.object(ex.urllib.request, "urlopen", side_effect=OSError("synthetic offline")):
            self.assertFalse(ex.notify("Synthetic notification test; never transmitted")["delivered_to_server"])

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory(prefix="exp4a_synthetic_") as directory:
            out = Path(directory)
            p = out / "sample.txt"
            p.write_text("a")
            ex.write_json(out / "ARTIFACT_MANIFEST.json", {"files": [{"filename": p.name, "bytes": 1, "sha256": ex.sha(p)}]})
            ex.verify_manifest(out)
            p.write_text("b")
            with self.assertRaises(RuntimeError):
                ex.verify_manifest(out)

    def test_full_pipeline_on_synthetic_data_only(self):
        """Exercise 200 bootstraps, every report/figure, reopening and hashing."""
        with tempfile.TemporaryDirectory(prefix="exp4a_synthetic_integration_") as directory:
            root = Path(directory)
            out = root / "synthetic_outputs"
            out.mkdir()
            raw = self.rng.binomial(3, .04, size=(43410, 28)).astype(np.int64)
            raw[:, 27] = 1
            counts = np.full(43410, 3, dtype=np.int64)
            labels = [list(np.flatnonzero(row[:27])) or [27] for row in raw]
            mapping = {e: ("Positive", "Negative", "Ambiguous")[i % 3] for i, e in enumerate(ex.LABELS)}
            def fake_audit(project, target):
                identity = {"emotion_column_order": ex.LABELS, "files": [], "note": "SYNTHETIC QA ONLY"}
                alignment = {"coverage": 1, "missing_ids": [], "matched_train_ids": 43410, "raw_train_rows": int(counts.sum())}
                leakage = {"used_raw_ids_subset_train": True, "used_raw_ids_intersection_dev": 0, "used_raw_ids_intersection_test": 0}
                for name, obj in (("data_identity.json", identity), ("train_raw_alignment_report.json", alignment), ("split_leakage_audit.json", leakage)):
                    ex.write_json(target / name, obj)
                ex.write_csv(target / "train_matrix_row_ids.csv", ({"row": i, "comment_id": f"synthetic_{i}"} for i in range(43410)))
                ex.write_csv(target / "train_raw_alignment.csv", [{"note": "synthetic"}])
                ex.write_json(target / "rater_count_statistics.json", {"distribution": {"3": 43410}})
                ex.write_csv(target / "rater_count_distribution.csv", [{"rater_count": 3, "comments": 43410}])
                (target / "DATA_AUDIT.md").write_text("SYNTHETIC QA ONLY\n")
                return raw, counts, counts, labels, identity, alignment, leakage, mapping
            with patch.object(ex, "audit", side_effect=fake_audit), patch.object(ex.socket, "gethostname", return_value="D3020-A3"), patch.object(ex, "git", return_value="synthetic"), redirect_stdout(io.StringIO()):
                result = ex.execute(root, out, audit_only=False)
                self.assertEqual(result["bootstrap_B"], 200)
                self.assertTrue((out / "SUCCESS.json").is_file())
                ex.verify_saved(out)
            self.assertFalse((out / "FAILURE.json").exists())


if __name__ == "__main__":
    unittest.main()
