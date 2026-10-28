"""Verify A1 predictions and create a unique matched-comparison evidence bundle."""
import argparse
import tarfile
import tempfile
from pathlib import Path

from a1lib import A0_RUN, run_path, sha, write
from a1run import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    root = run_path(parser.parse_args().run_root)
    verify(root)
    dest = Path(tempfile.mkdtemp(prefix="exp6c1_review_", dir=root.parent))
    archive = dest / "Exp6C1_review.tar.gz"
    files = [p for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".md", ".csv", ".npz", ".npy", ".txt"}]
    with tarfile.open(archive, "w:gz") as t:
        for p in sorted(files):
            t.add(p, arcname=root.name + "/" + p.name, recursive=False)
        for p in sorted((root / "smoke").glob("*.json")):
            t.add(p, arcname=root.name + "/smoke/" + p.name, recursive=False)
        for name in ["test_metrics.json", "test_per_label_metrics.csv", "test_predictions.npz", "SUCCESS.json", "config.json", "checkpoint_selection.json"]:
            t.add(A0_RUN / name, arcname="A0_reference/" + name, recursive=False)
    write(dest / "REVIEW_BUNDLE.json", dict(archive_sha256=sha(archive), run_root=str(root),
        contains_checkpoint=False, original_A0_readonly=True,
        note="This bundle is a subset; full A0/A1 seals refer to complete original Ubuntu run directories."))
    print("REVIEW_ARCHIVE", archive)
    print("SHA256", sha(archive))
    print("MAC_DOWNLOAD_COMMAND: scp ubuntu4090:" + str(archive) + " ~/Desktop/PSP303-FYP/交付/Exp6C1_RoBERTa27_HypAux_20261020/")


if __name__ == "__main__":
    main()
