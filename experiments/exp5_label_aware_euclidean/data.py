"""Canonical GoEmotions TSV loading with exact split/order/label-signature gates."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from exp5lib import ASSET_MANIFEST, DATA_DIR, objsha, read, require, sha

LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
SPLIT_SIZES = {"train": 43410, "validation": 5426, "test": 5427}
OFFICIAL_HASHES = {
    "train.tsv": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
    "dev.tsv": "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62",
    "test.tsv": "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
}
RARE_IDS = [16, 21, 23, 19, 12, 24, 14]


def multi_hot(labels):
    require(labels and len(labels) == len(set(labels)), f"Empty or duplicate labels: {labels}")
    require(all(type(label) is int and 0 <= label < 28 for label in labels), f"Invalid labels: {labels}")
    target = np.zeros(28, dtype=np.float32)
    target[labels] = 1.0
    return target


def load_records(split):
    require(split in SPLIT_FILES, f"Unknown split: {split}")
    path = DATA_DIR / SPLIT_FILES[split]
    require(path.is_file(), f"Canonical split missing: {path}")
    require(sha(path) == OFFICIAL_HASHES[path.name], f"Canonical TSV checksum mismatch: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_MINIMAL))
    require(len(rows) == SPLIT_SIZES[split], f"Wrong {split} size")
    records = []
    for index, row in enumerate(rows):
        require(len(row) == 3, f"Malformed {split} row {index + 1}")
        record = {"text": row[0], "labels": [int(x) for x in row[1].split(",")], "id": row[2]}
        multi_hot(record["labels"])
        records.append(record)
    require(len({record["id"] for record in records}) == len(records), f"Duplicate IDs in {split}")
    return records


def signature(records):
    rows, gold = hashlib.sha256(), hashlib.sha256()
    ids = []
    for record in records:
        row = {key: record[key] for key in ("id", "text", "labels")}
        rows.update((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
        ids.append(record["id"])
        gold.update(multi_hot(record["labels"]).astype(np.uint8).tobytes())
    return {"rows_sha256": rows.hexdigest(), "ids_sha256": objsha(ids), "gold_sha256": gold.hexdigest(), "size": len(ids)}


def load_all_verified():
    manifest = read(ASSET_MANIFEST)
    require(manifest["status"] == "GOEMOTIONS_ASSETS_READY", "Prepared asset manifest is not ready")
    require(manifest["split_sizes"] == SPLIT_SIZES, "Prepared split sizes differ")
    require(manifest["rare_ids"] == RARE_IDS, "Rare-label definition differs")
    result, seen = {}, set()
    for split in SPLIT_FILES:
        result[split] = load_records(split)
        ids = {record["id"] for record in result[split]}
        require(not seen.intersection(ids), "Comment IDs leak across splits")
        seen.update(ids)
        require(signature(result[split]) == manifest["dataset_signatures"][split], f"{split} signature differs from prepared data")
    return result, manifest


def frequencies(records):
    counts = np.zeros(28, dtype=np.int64)
    for record in records:
        counts += multi_hot(record["labels"]).astype(np.int64)
    derived = sorted(range(28), key=lambda i: (int(counts[i]), i))[: math.ceil(28 * 0.25)]
    require(derived == RARE_IDS, "Rare-label IDs changed")
    return counts


class RecordDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        encoded = self.tokenizer(
            [example["text"] for example in examples],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            np.stack([multi_hot(example["labels"]) for example in examples]), dtype=torch.float32
        )
        encoded["record_ids"] = [example["id"] for example in examples]
        return encoded
