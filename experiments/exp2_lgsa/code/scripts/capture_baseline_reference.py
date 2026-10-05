"""Read the prior archive; copy small verified reference artifacts only."""
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT.parent / "goemotions/backups/goemotions_bert_seed42_job2968116_20260921_141855.tar.gz"
prefix = "outputs/bert_baseline/seed42/"
names = ["test_metrics.json", "test_per_label_fixed_0.30.csv", "rare_label_metrics.json",
         "dataset_analysis/rare_labels.json", "dataset_analysis/label_frequencies.csv",
         "config.json", "environment.json", "threshold_selection.json"]
dest = ROOT / "references/baseline"
dest.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as tar:
    success = json.load(tar.extractfile(prefix + "SUCCESS.json"))
    artifacts = {}
    for name in names:
        data = tar.extractfile(prefix + name).read()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == success["artifacts_sha256"][name], name
        output = dest / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        artifacts[name] = digest
metrics = json.loads((dest / "test_metrics.json").read_text())
rare_ids = set(json.loads((dest / "dataset_analysis/rare_labels.json").read_text())["label_ids"])
rows = list(csv.DictReader(io.StringIO((dest / "test_per_label_fixed_0.30.csv").read_text())))
rare_f1 = sum(float(r["f1"]) for r in rows if int(r["label_id"]) in rare_ids) / len(rare_ids)
other_f1 = sum(float(r["f1"]) for r in rows if int(r["label_id"]) not in rare_ids) / (28-len(rare_ids))
record = {"experiment_name": "Standard BERT GoEmotions Baseline", "fyp_experiment_number": 1,
          "archive": str(archive), "slurm_job_id_from_archive": success["slurm_job_id"],
          "artifact_integrity": "8 copied files match archived SUCCESS.json SHA256",
          "live_sacct_verified_this_session": False, "metrics": metrics,
          "rare_f1_fixed_030": rare_f1, "non_rare_f1_fixed_030": other_f1,
          "artifacts_sha256": artifacts}
(dest / "reference_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"micro_f1": metrics["fixed_0.30"]["micro_f1"],
                  "macro_f1": metrics["fixed_0.30"]["macro_f1"],
                  "rare_f1": rare_f1, "non_rare_f1": other_f1,
                  "copied_and_verified": len(artifacts)}, indent=2))
