"""Canonical official TSVs; remove neutral by name, never drop mixed positives."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from common import REPO, object_sha, require, sha, write

OFFICIAL = {
    "train": ("train.tsv", 43410, "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5"),
    "validation": ("dev.tsv", 5426, "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62"),
    "test": ("test.tsv", 5427, "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4"),
}
MAPPING_SHA = "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89"


def filter_records(records, canonical_names):
    require(len(canonical_names) == 28 and len(set(canonical_names)) == 28, "Need canonical 28 labels")
    neutral = canonical_names.index("neutral")
    old = [i for i in range(28) if i != neutral]
    mapping = {v: k for k, v in enumerate(old)}
    kept, excluded = [], []
    for r in records:
        require(r["labels"] and len(set(r["labels"])) == len(r["labels"]), "Empty/duplicate source labels")
        require(all(type(i) is int and 0 <= i < 28 for i in r["labels"]), "Source label ID invalid")
        ids = [mapping[i] for i in r["labels"] if i != neutral]
        if ids:
            kept.append(dict(id=r["id"], text=r["text"], labels=ids,
                             original_labels=list(r["labels"])))
        else:
            require(r["labels"] == [neutral], "Only neutral-only records may be removed")
            excluded.append(r)
    return kept, excluded, old


def targets(records):
    result = np.zeros((len(records), 27), dtype=np.float32)
    for i, row in enumerate(records):
        require(row["labels"] and len(set(row["labels"])) == len(row["labels"]), "Invalid retained label set")
        require(all(type(v) is int and 0 <= v < 27 for v in row["labels"]), "Invalid new index")
        result[i, row["labels"]] = 1
    require(np.all(result.sum(1) >= 1), "Empty retained targets")
    return result


def load_dataset(repo=REPO):
    base = Path(repo) / "数据集/GoEmotions"
    mapping_path = base / "metadata/emotions.txt"
    require(sha(mapping_path) == MAPPING_SHA, "Canonical label mapping checksum differs")
    names28 = mapping_path.read_text().splitlines()
    names27 = [n for n in names28 if n != "neutral"]
    result, manifest, seen, texts = {}, {}, set(), {}
    files = {str(mapping_path): sha(mapping_path)}
    for split, (name, count, checksum) in OFFICIAL.items():
        path = base / "processed" / name
        require(sha(path) == checksum, f"Official split differs: {split}")
        files[str(path)] = checksum
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL))
        require(len(rows) == count and all(len(r) == 3 for r in rows), f"Malformed official {split}")
        original = [dict(text=r[0], labels=[int(v) for v in r[1].split(",")], id=r[2]) for r in rows]
        ids = [r["id"] for r in original]
        require(all(ids) and len(ids) == len(set(ids)), "Missing/duplicate canonical IDs")
        require(not seen.intersection(ids), "Canonical IDs cross splits")
        seen.update(ids)
        kept, excluded, old = filter_records(original, names28)
        gold = targets(kept)
        # Filtering preserves byte-identical text, original label order and row order.
        by_id = {r["id"]: r for r in original}
        for r in kept:
            source = by_id[r["id"]]
            require(r["text"] == source["text"], "Text modified")
            require([old[i] for i in r["labels"]] == [i for i in source["labels"] if names28[i] != "neutral"], "Labels modified")
        cardinality = gold.sum(1)
        manifest[split] = dict(original_count=len(original), retained_count=len(kept),
            removed_neutral_only=len(excluded), single_emotion=int((cardinality == 1).sum()),
            multi_emotion=int((cardinality > 1).sum()), multi_fraction=float((cardinality > 1).mean()),
            mean_positive_labels=float(cardinality.mean()), positive_support=gold.sum(0).astype(int).tolist(),
            retained_neutral_plus_emotion=sum(names28.index("neutral") in r["original_labels"] for r in kept),
            original_ids=ids, retained_ids=[r["id"] for r in kept], excluded_neutral_only_ids=[r["id"] for r in excluded],
            original_rows_sha256=object_sha(original), retained_rows_sha256=object_sha(kept),
            targets_sha256=__import__("hashlib").sha256(gold.astype(np.uint8).tobytes()).hexdigest(),
            source_tsv_sha256=checksum)
        result[split] = kept
        texts[split] = {r["text"] for r in original}
    overlaps = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        filtered_a = {r["text"] for r in result[a]}
        filtered_b = {r["text"] for r in result[b]}
        require(filtered_a.intersection(filtered_b) <= texts[a].intersection(texts[b]), "New text leakage")
        overlaps[a + "_" + b] = dict(original_exact_text_overlap=len(texts[a] & texts[b]),
                                     retained_exact_text_overlap=len(filtered_a & filtered_b))
    return result, dict(protocol="official simplified splits; >=1 non-neutral positive; 27 multi-hot",
        canonical_names28=names28, names27=names27, neutral_original_index=names28.index("neutral"),
        mapping=[dict(new_index=i, original_index=old[i], emotion=n) for i, n in enumerate(names27)],
        source_files_sha256=files, splits=manifest, canonical_id_overlap=0,
        text_overlap_audit=overlaps, text_overlap_note="Existing duplicate text, if any, is inherited from official splits; no deduplication or resplitting.")


def save_audit(root, manifest):
    for name in ["DATASET_MANIFEST.json", "dataset_manifest.json"]:
        write(Path(root) / name, manifest)
    lines = ["# Exp6C-0 dataset audit", "", manifest["protocol"], "",
             "Canonical mapping is read and SHA-256 checked from metadata/emotions.txt; neutral is removed by name.",
             "Original/retained/excluded canonical IDs and row/target hashes are in DATASET_MANIFEST.json.", "",
             "| Split | Original | Retained | Neutral-only removed | Single | Multi | Multi fraction | Mean positives | Mixed neutral retained |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, s in manifest["splits"].items():
        lines.append(f"| {split} | {s['original_count']} | {s['retained_count']} | {s['removed_neutral_only']} | {s['single_emotion']} | {s['multi_emotion']} | {s['multi_fraction']:.6f} | {s['mean_positive_labels']:.6f} | {s['retained_neutral_plus_emotion']} |")
    lines += ["", "| Emotion | New ID | Canonical ID | Train support | Validation support | Test support |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in manifest["mapping"]:
        counts = [str(manifest["splits"][s]["positive_support"][r["new_index"]]) for s in OFFICIAL]
        lines.append(f"| {r['emotion']} | {r['new_index']} | {r['original_index']} | " + " | ".join(counts) + " |")
    lines += ["", "Canonical IDs are unique within and disjoint across splits. Retained text and labels match the source exactly after neutral removal.",
              "All retained targets are 27-dimensional multi-hot vectors with >=1 positive. Multi-label examples remain multi-label.",
              manifest["text_overlap_note"], "", json_text(manifest["text_overlap_audit"]), "",
              "This is neither the original HypEmo single-label protocol nor the earlier full 28-label protocol."]
    (Path(root) / "DATASET_AUDIT.md").write_text("\n".join(lines) + "\n")


def json_text(value):
    import json
    return "```json\n" + json.dumps(value, indent=2) + "\n```"


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        import torch
        x = self.tokenizer([r["text"] for r in rows], truncation=True, max_length=50,
                           padding=True, return_tensors="pt", return_token_type_ids=False)
        x["labels"] = torch.from_numpy(targets(rows))
        x["ids"] = [r["id"] for r in rows]
        return x


def loader(records, tokenizer, shuffle=False):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(records, batch_size=16 if shuffle else 64, shuffle=shuffle,
                      collate_fn=Collator(tokenizer), num_workers=0, pin_memory=True,
                      generator=torch.Generator().manual_seed(42) if shuffle else None)
