"""HPC first-time download; resolve mutable Hub revisions once, then run offline."""
import argparse
import shutil
from pathlib import Path

from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from src.common import ROOT, load_config, read_json, sha256, utc_now, write_json
from src.data import OFFICIAL_HASHES, SPLIT_SIZES, analyze, load_official_tsv, load_prepared, validate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=("huggingface", "official-tsv"), default="huggingface")
    args = parser.parse_args()
    config = load_config()
    assets = ROOT / "assets"
    manifest_path = assets / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["data_source"] != args.data_source:
            raise RuntimeError("Prepared data source differs. Use the same --data-source; do not silently replace assets.")
        load_prepared()
        print("Assets already prepared; reusing pinned manifest.", flush=True)
        return
    assets.mkdir(exist_ok=True)
    model_sha = HfApi().model_info(config["model_repo"], revision=config["model_revision"]).sha
    model_path = snapshot_download(
        config["model_repo"], revision=model_sha,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                        "vocab.txt", "special_tokens_map.json"])
    for required in ("config.json", "model.safetensors", "vocab.txt"):
        if not (Path(model_path) / required).is_file():
            raise RuntimeError(f"Missing model asset: {required}")
    AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.data_source == "huggingface":
        dataset_sha = HfApi().dataset_info(config["dataset_name"], revision=config["dataset_revision"]).sha
        dataset = load_dataset(config["dataset_name"], config["dataset_config"], revision=dataset_sha)
        source = {"repository": config["dataset_name"], "configuration": config["dataset_config"], "revision": dataset_sha}
    else:
        dataset = load_official_tsv(ROOT / "data/official")
        source = {"repository": "google-research/google-research/goemotions/data", "configuration": "official predefined TSV splits",
                  "sha256": OFFICIAL_HASHES, "deviation": "Explicit official TSV source instead of Hugging Face simplified; same official splits/taxonomy."}
    validate_dataset(dataset)
    # Resume interrupted preparation only in its dedicated temporary directory.
    temporary = assets / "dataset.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    dataset.save_to_disk(str(temporary))
    destination = assets / "dataset"
    if destination.exists():
        raise RuntimeError("An unmanifested assets/dataset exists; inspect it before retrying.")
    temporary.rename(destination)
    disk = load_from_disk(str(destination))
    manifest = {"created_at": utc_now(), "data_source": args.data_source, "dataset": source,
                "split_sizes": SPLIT_SIZES, "disk_fingerprints": {s: disk[s]._fingerprint for s in disk},
                "model_repo": config["model_repo"], "model_revision": model_sha, "model_snapshot": model_path,
                "model_sha256": {p.name: sha256(p) for p in sorted(Path(model_path).iterdir()) if p.is_file()}}
    write_json(manifest_path, manifest)
    analyze(disk, ROOT / "outputs/dataset_analysis")
    print(f"Prepared official splits {SPLIT_SIZES}; BERT revision {model_sha}", flush=True)


if __name__ == "__main__":
    main()
