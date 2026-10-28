"""CPU-verify a successful run and create a small unique evidence bundle."""
import argparse
import tempfile
import tarfile
from pathlib import Path

from common import read, run_path, sha, write
from run import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = run_path(args.run_root)
    verify(root)
    destination = Path(tempfile.mkdtemp(prefix="exp6c0_review_", dir=root.parent))
    files = [p for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".md", ".csv", ".txt"}]
    # Full prediction arrays are compact and necessary for independent future audits.
    files += [root / "test_predictions.npz", root / "validation_predictions.npz"]
    archive = destination / "Exp6C0_review.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(files):
            tar.add(p, arcname=root.name + "/" + p.name, recursive=False)
    write(destination / "REVIEW_BUNDLE.json", dict(run_root=str(root), archive=str(archive),
        archive_sha256=sha(archive), contains_checkpoint=False,
        files_sha256={p.name: sha(p) for p in files},
        note="Use original Ubuntu SUCCESS.json to verify the complete run; this is a subset for review."))
    print("REVIEW_ARCHIVE", archive)
    print("SHA256", sha(archive))
    print("MAC_DOWNLOAD_COMMAND: scp ubuntu4090:" + str(archive) + " ~/Desktop/PSP303-FYP/交付/Exp6C0_RoBERTa27_20261020/")


if __name__ == "__main__":
    main()
