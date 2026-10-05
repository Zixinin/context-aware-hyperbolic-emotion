"""Official splits, validated taxonomy and float32 multi-hot targets."""
import csv
import math
from pathlib import Path

import numpy as np

from .common import ROOT, read_json, sha256, write_json

LABEL_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
SPLIT_SIZES = {"train": 43410, "validation": 5426, "test": 5427}
OFFICIAL_HASHES = {
    "train.tsv": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
    "dev.tsv": "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62",
    "test.tsv": "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
    "emotions.txt": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
}


def multi_hot(label_ids):
    vector = np.zeros(28, dtype=np.float32)
    if not label_ids or len(set(label_ids)) != len(label_ids):
        raise ValueError(f"Empty or duplicated labels: {label_ids}")
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= 28 for i in label_ids):
        raise ValueError(f"Invalid label IDs: {label_ids}")
    vector[label_ids] = 1.0
    return vector


def load_official_tsv(directory):
    from datasets import Dataset, DatasetDict
    directory = Path(directory)
    for name, expected in OFFICIAL_HASHES.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Official TSV checksum mismatch: {name}")
    if (directory / "emotions.txt").read_text().splitlines() != LABEL_NAMES:
        raise ValueError("Label order differs from official taxonomy.")
    result = {}
    for split, filename in (("train", "train.tsv"), ("validation", "dev.tsv"), ("test", "test.tsv")):
        with (directory / filename).open(encoding="utf-8", newline="") as stream:
            records = [{"text": text, "labels": [int(i) for i in labels.split(",")], "id": row_id}
                       for text, labels, row_id in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)]
        result[split] = Dataset.from_list(records)
    return DatasetDict(result)


def validate_dataset(dataset):
    if set(dataset) != set(SPLIT_SIZES):
        raise ValueError("Expected official train/validation/test splits.")
    seen = set()
    for split, size in SPLIT_SIZES.items():
        if len(dataset[split]) != size:
            raise ValueError(f"Wrong {split} size: {len(dataset[split])}, expected {size}")
        feature = dataset[split].features["labels"]
        names = getattr(getattr(feature, "feature", None), "names", None)
        if names is not None and names != LABEL_NAMES:
            raise ValueError("Hugging Face label order differs from canonical taxonomy.")
        ids = dataset[split]["id"]
        if len(set(ids)) != size or seen.intersection(ids):
            raise ValueError("Duplicate comment IDs inside/across official splits.")
        seen.update(ids)
        for labels in dataset[split]["labels"]:
            multi_hot(labels)


def load_prepared():
    from datasets import load_from_disk
    manifest = read_json(ROOT / "assets/manifest.json")
    dataset = load_from_disk(str(ROOT / "assets/dataset"))
    validate_dataset(dataset)
    current = {split: dataset[split]._fingerprint for split in dataset}
    if current != manifest["disk_fingerprints"]:
        raise ValueError("Cached dataset fingerprints differ from prepared manifest.")
    for name, expected in manifest["model_sha256"].items():
        if sha256(Path(manifest["model_snapshot"]) / name) != expected:
            raise ValueError(f"Cached BERT file checksum mismatch: {name}")
    return dataset, manifest


def analyze(dataset, output_dir):
    """Rare set is defined solely by all 43,410 TRAIN examples, also for smoke."""
    import pandas as pd
    from .visualize import frequency_chart
    validate_dataset(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, statistics = {}, {}
    for split in SPLIT_SIZES:
        targets = np.stack([multi_hot(labels) for labels in dataset[split]["labels"]])
        cardinality = targets.sum(axis=1)
        counts[split] = targets.sum(axis=0).astype(int)
        statistics[split] = {
            "num_examples": len(targets), "average_labels_per_example": float(cardinality.mean()),
            "single_label_percentage": float(100 * (cardinality == 1).mean()),
            "multi_label_percentage": float(100 * (cardinality > 1).mean()),
            "min_labels": int(cardinality.min()), "max_labels": int(cardinality.max()),
        }
    rare_ids = sorted(range(28), key=lambda i: (int(counts["train"][i]), i))[:math.ceil(28 * 0.25)]
    rare = {"rule": "bottom 25% of 28 labels by full training frequency; ties: label_id ascending",
            "label_ids": rare_ids, "label_names": [LABEL_NAMES[i] for i in rare_ids],
            "training_counts": [int(counts["train"][i]) for i in rare_ids]}
    frame = pd.DataFrame({"label_id": range(28), "label": LABEL_NAMES,
                          **{f"{split}_count": counts[split] for split in SPLIT_SIZES},
                          "is_rare": [i in rare_ids for i in range(28)]})
    frame.to_csv(output_dir / "label_frequencies.csv", index=False)
    write_json(output_dir / "dataset_statistics.json", {"splits": statistics, "labels": dict(enumerate(LABEL_NAMES))})
    write_json(output_dir / "rare_labels.json", rare)
    frequency_chart(frame, output_dir / "label_frequency.png")
    return frame, statistics, rare


def tokenize_dataset(dataset, tokenizer, max_length):
    def encode(batch):
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        tokenized["labels"] = [multi_hot(ids).tolist() for ids in batch["labels"]]
        return tokenized
    return dataset.map(encode, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")


class MultiLabelCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        import torch
        labels = torch.tensor([example["labels"] for example in examples], dtype=torch.float32)
        batch = self.tokenizer.pad([{k: v for k, v in example.items() if k != "labels"}
                                   for example in examples], padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch
