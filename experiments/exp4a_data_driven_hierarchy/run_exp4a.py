#!/usr/bin/env python3
"""CPU-only, TRAIN-only Exp4A; formal bootstrap count/seed are fixed."""
import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

B = 200
SEED = 2026
KS = range(2, 11)
TOPIC = "https://ntfy.sh/zixin-fyp-4090-1119"
EXPECTED_HASHES = {
    "processed/train.tsv": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
    "processed/dev.tsv": "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62",
    "processed/test.tsv": "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
    "metadata/emotions.txt": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
    "metadata/sentiment_dict.json": "801f6f028082141bd38541a4b14d65b33bdc55d29c7b2aaa029fefecf206f76b",
    "raw/goemotions_1.csv": "cac049036bad5d68d1081f72b65f2cc51e4df82af05e3e22cfa747051cac1af3",
    "raw/goemotions_2.csv": "f699ecc5aa425c1720c1d02475f1e41815244b680bd75b282eb770d2c76cd84d",
    "raw/goemotions_3.csv": "467f1e7191af00f2e76cc7f425885c2dc304bea8aff284b10e8c460d22f2e1af",
}
LABELS = "admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise".split()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def notify(message):
    try:
        req = urllib.request.Request(TOPIC, data=message.encode("utf-8"), method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read())
        return {"attempted": True, "delivered_to_server": True, "id": payload.get("id")}
    except Exception as exc:
        print("NTFY_WARNING:", repr(exc), flush=True)
        return {"attempted": True, "delivered_to_server": False, "error": repr(exc)}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_science():
    global np, hierarchy, distance, metrics, plt
    import numpy as np
    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    from sklearn import metrics
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return {name: importlib.import_module(name).__version__ for name in ("numpy", "scipy", "sklearn", "matplotlib")}


def read_split(path, parse_train=False):
    ids, texts, annotations = [], [], []
    with path.open(encoding="utf-8", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            require(len(row) == 3, f"{path}:{line}: expected three columns")
            require(bool(row[2]) and row[2] == row[2].strip(), f"Invalid comment ID at {path}:{line}")
            ids.append(row[2])
            if parse_train:
                values = [int(x) for x in row[1].split(",")]
                require(values and len(values) == len(set(values)) and all(0 <= x < 28 for x in values), "Invalid TRAIN label IDs")
                annotations.append(values)
                texts.append(row[0])
    require(len(ids) == len(set(ids)), f"Duplicate IDs in {path}")
    return ids, texts, annotations


def audit(root, out):
    data = root / "数据集/GoEmotions"
    baseline = root / "experiments/exp1_bert_baseline/goemotions/data/official"
    identities = []
    for rel, expected in EXPECTED_HASHES.items():
        path = data / rel
        require(path.is_file(), f"Missing official source: {path}")
        digest = sha(path)
        require(digest == expected, f"Source hash mismatch: {path}")
        if rel.startswith("raw/"):
            url = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/" + path.name
        else:
            url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/" + path.name
        identities.append({"path": str(path), "filename": path.name, "bytes": path.stat().st_size,
                           "sha256": digest, "source_url": url, "audited_at": timestamp(),
                           "download_date": None, "download_date_note": "Existing local official files; original download date not independently established."})
        if rel.startswith("processed/") or rel == "metadata/emotions.txt":
            old = baseline / path.name
            require(old.is_file() and sha(old) == digest, f"Prior baseline identity mismatch: {old}")
    emotions = (data / "metadata/emotions.txt").read_text().splitlines()
    require(emotions == LABELS + ["neutral"], "Canonical label order mismatch")
    split_ids = {}
    for name, count in (("train", 43410), ("dev", 5426), ("test", 5427)):
        ids, texts, annotations = read_split(data / "processed" / f"{name}.tsv", name == "train")
        require(len(ids) == count, f"Unexpected {name} size")
        split_ids[name] = set(ids)
        if name == "train":
            train_ids, train_texts, train_labels = ids, texts, annotations
    overlap = {f"{a}_{b}": len(split_ids[a] & split_ids[b]) for a, b in combinations(split_ids, 2)}
    require(not any(overlap.values()), "Canonical splits overlap")
    index = {comment_id: i for i, comment_id in enumerate(train_ids)}
    sums = np.zeros((43410, 28), dtype=np.int64)
    counts = np.zeros(43410, dtype=np.int64)
    labeled_counts = np.zeros(43410, dtype=np.int64)
    unclear_counts = np.zeros(43410, dtype=np.int64)
    zero_counts = np.zeros(43410, dtype=np.int64)
    seen = set()
    used_ids = set()
    schemas = []
    for filename in ("goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"):
        n_all = n_train = 0
        with (data / "raw" / filename).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            require({"id", "rater_id", "text", "example_very_unclear", *emotions}.issubset(reader.fieldnames), "Raw schema mismatch")
            for row in reader:
                n_all += 1
                comment_id = row["id"]
                if comment_id not in index:
                    continue  # Evaluation judgments never parsed or accumulated.
                i = index[comment_id]
                require(bool(row["rater_id"]), "Empty TRAIN rater ID")
                pair = (comment_id, row["rater_id"])
                require(pair not in seen, "Duplicate TRAIN comment/rater pair")
                seen.add(pair)
                require(row["text"] == train_texts[i], f"TRAIN text identity mismatch for {comment_id}")
                require(all(row[e] in {"0", "1"} for e in emotions), "Nonbinary TRAIN raw judgment")
                values = np.array([int(row[e]) for e in emotions], dtype=np.int64)
                flag = row["example_very_unclear"].lower()
                require(flag in {"true", "false", "0", "1"}, "Invalid unclear flag")
                unclear = flag in {"true", "1"}
                require(not unclear or values.sum() == 0, "Unclear row unexpectedly has an emotion judgment")
                sums[i] += values
                counts[i] += 1
                labeled_counts[i] += int(values.sum() > 0)
                zero_counts[i] += int(values.sum() == 0)
                unclear_counts[i] += int(unclear)
                used_ids.add(comment_id)
                n_train += 1
            schemas.append({"filename": filename, "columns": reader.fieldnames, "all_rows": n_all, "train_rows": n_train})
    missing = sorted(split_ids["train"] - used_ids)
    alignment = {"expected_train_ids": 43410, "matched_train_ids": len(used_ids), "missing_ids": missing,
                 "coverage": len(used_ids) / 43410, "duplicate_comment_rater_pairs": 0,
                 "raw_train_rows": int(counts.sum()), "raw_schema": schemas,
                 "unclear_rows": int(unclear_counts.sum()), "zero_label_rows": int(zero_counts.sum()),
                 "unit": "canonical TRAIN comment; all rater rows retained for PRIMARY"}
    write_json(out / "train_raw_alignment_report.json", alignment)
    leakage = {"split_id_overlaps": overlap, "used_raw_ids_subset_train": used_ids <= split_ids["train"],
               "used_raw_ids_intersection_dev": len(used_ids & split_ids["dev"]),
               "used_raw_ids_intersection_test": len(used_ids & split_ids["test"]),
               "evaluation_label_fields_parsed": False, "used_id_count": len(used_ids)}
    write_json(out / "split_leakage_audit.json", leakage)
    require(not missing and np.all(counts > 0), "Incomplete TRAIN alignment: stop")
    require(leakage["used_raw_ids_subset_train"] and not leakage["used_raw_ids_intersection_dev"] and not leakage["used_raw_ids_intersection_test"], "Leakage: stop")
    require(np.all(labeled_counts > 0), "Official-filter sensitivity would lose TRAIN comments: stop")
    write_csv(out / "train_raw_alignment.csv", ({"train_row": i, "comment_id": cid, "rater_count": int(counts[i]),
              "labeled_rater_count": int(labeled_counts[i]), "unclear_count": int(unclear_counts[i]),
              "zero_label_count": int(zero_counts[i])} for i, cid in enumerate(train_ids)))
    write_csv(out / "train_matrix_row_ids.csv", ({"row": i, "comment_id": cid} for i, cid in enumerate(train_ids)))
    old_path = root / "experiments/exp2_lgsa/code/config/goemotions_hierarchy_v0.json"
    frozen_path = root / "experiments/exp2_1_multiseed/frozen/hierarchy.json"
    old = read_json(old_path)
    require(sha(old_path) == sha(frozen_path), "Hierarchy v0 differs from frozen Exp2.1 file")
    require(len(old["labels"]) == 28, "Unexpected hierarchy v0 label count")
    mapping = {row["fine_label"]: row["coarse_label"] for row in old["labels"]}
    require(set(mapping) == set(emotions), "Hierarchy v0 label set mismatch")
    require(all(mapping[e] in {"Positive", "Negative", "Ambiguous"} for e in LABELS), "Unexpected v0 emotion group")
    sentiment = read_json(data / "metadata/sentiment_dict.json")
    require(all(e in sentiment[mapping[e].lower()] for e in LABELS), "v0/source sentiment disagreement")
    identity = {"files": identities, "train_count": 43410, "label_ids": dict(enumerate(emotions)),
                "emotion_column_order": LABELS, "neutral_id": 27,
                "v0_file": str(old_path), "v0_sha256": sha(old_path), "frozen_v0_sha256": sha(frozen_path),
                "matrix_row_order": "canonical train.tsv order"}
    write_json(out / "data_identity.json", identity)
    distribution = Counter(map(int, counts))
    stats = {"min": int(counts.min()), "max": int(counts.max()), "mean": float(counts.mean()),
             "median": float(np.median(counts)), "distribution": dict(sorted(distribution.items())),
             "one_rater": int((counts == 1).sum()), "two_raters": int((counts == 2).sum()),
             "three_raters": int((counts == 3).sum()), "four_or_more": int((counts >= 4).sum()),
             "definition": "all unique raters, including no-label/unclear rows"}
    write_json(out / "rater_count_statistics.json", stats)
    write_csv(out / "rater_count_distribution.csv", ({"rater_count": k, "comments": v} for k, v in sorted(distribution.items())))
    text = (f"# Exp4A data audit\n\nTRAIN: 43410 unique canonical IDs; dev: 5426; test: 5427.\n"
            f"Alignment: {len(used_ids)}/43410; raw TRAIN rows: {counts.sum()}; duplicate comment/rater pairs: 0.\n"
            f"All TRAIN texts match raw records. Split overlaps and used-ID evaluation intersections are zero.\n"
            f"Neutral ID: 27; matrix emotion order: {', '.join(LABELS)}.\n"
            f"Unclear rows: {unclear_counts.sum()}; no-label rows: {zero_counts.sum()}. PRIMARY retains all rows per specification.\n"
            "Official-filter sensitivity excludes no-label rows, including unclear rows, without losing any TRAIN comments.\n"
            "Baseline split and emotion-list hashes match the canonical inputs. v0 matches frozen Exp2.1.\n"
            "Official source URLs and expected/matched SHA256 hashes are in data_identity.json. Original download date is unknown.\n")
    (out / "DATA_AUDIT.md").write_text(text, encoding="utf-8")
    return sums, counts, labeled_counts, train_labels, identity, alignment, leakage, mapping


def correlations(matrix):
    require(matrix.ndim == 2 and matrix.shape[1] == 27 and np.isfinite(matrix).all(), "Invalid analysis matrix")
    require(np.all(matrix.var(axis=0) > 0), "Zero-variance emotion: stop; no NaN substitution")
    r = np.corrcoef(matrix, rowvar=False)
    require(np.isfinite(r).all() and np.allclose(r, r.T, atol=1e-12) and np.allclose(np.diag(r), 1), "Invalid Pearson matrix")
    require(r.min() >= -1 - 1e-12 and r.max() <= 1 + 1e-12, "Pearson out of bounds")
    # Only floating-point roundoff is clipped; invalid correlations already fail.
    r = np.clip((r + r.T) / 2, -1, 1)
    np.fill_diagonal(r, 1)
    d = 1 - r
    np.fill_diagonal(d, 0)
    require(np.isfinite(d).all() and np.allclose(d, d.T) and np.all(d >= 0), "Invalid distance")
    return r, d


def linkage_from(d, method="ward"):
    z = hierarchy.linkage(distance.squareform(d, checks=True), method=method, optimal_ordering=True)
    require(hierarchy.is_valid_linkage(z) and np.isfinite(z).all() and hierarchy.is_monotonic(z), "Invalid/nonmonotonic linkage")
    return z


def cut(z, k):
    labels = hierarchy.cut_tree(z, n_clusters=[k]).ravel()
    require(len(set(labels)) == k, "Exact-k cut failed")
    return labels


def clades(z):
    nodes = {i: frozenset([LABELS[i]]) for i in range(27)}
    result = []
    for i, row in enumerate(z):
        leaves = nodes[int(row[0])] | nodes[int(row[1])]
        nodes[27 + i] = leaves
        result.append(leaves)
    return result


def tree(z, prefix, support=None):
    nodes = {i: {"node_id": LABELS[i], "label_id": i, "leaf_count": 1, "leaf_labels": [LABELS[i]], "children": []} for i in range(27)}
    for i, row in enumerate(z):
        a, b = nodes[int(row[0])], nodes[int(row[1])]
        leaves = sorted(a["leaf_labels"] + b["leaf_labels"])
        node = {"node_id": f"{prefix}_cluster_{i + 1:03d}", "merge_distance": float(row[2]),
                "leaf_count": len(leaves), "leaf_labels": leaves, "children": [a, b]}
        if support is not None:
            node["bootstrap_support"] = support[frozenset(leaves)]
            node["stable_70"] = node["bootstrap_support"] >= .70
        nodes[27 + i] = node
    return {"schema_version": 1, "source": "TRAIN_ONLY_RAW_RATER_AVERAGED" if prefix == "raw" else "TRAIN_ONLY_SIMPLIFIED_BINARY",
            "neutral_excluded_from_clustering": True, "root": {"node_id": "ROOT", "children": [
                {"node_id": "EMOTION", "children": [nodes[52]]},
                {"node_id": "NEUTRAL", "children": [{"node_id": "neutral", "label_id": 27, "children": []}]}]}}


def validate_tree(obj):
    def walk(node):
        children = node.get("children", [])
        if not children:
            return [node["node_id"]]
        leaves = sum((walk(child) for child in children), [])
        if "leaf_labels" in node:
            require(sorted(leaves) == node["leaf_labels"] and len(leaves) == node["leaf_count"], "Tree clade metadata mismatch")
            require(len(children) == 2, "Clustering node must be binary")
        return leaves
    root = obj["root"]
    require([x["node_id"] for x in root["children"]] == ["EMOTION", "NEUTRAL"], "Root structure mismatch")
    leaves = walk(root)
    require(Counter(leaves) == Counter(LABELS + ["neutral"]), "Final tree leaf coverage mismatch")
    emotion_leaves = walk(root["children"][0])
    require(Counter(emotion_leaves) == Counter(LABELS), "Emotion tree contains incorrect leaves")


def matrix_csv(out, name, matrix):
    write_csv(out / f"{name}.csv", ({"emotion": label, **{e: float(matrix[i, j]) for j, e in enumerate(LABELS)}} for i, label in enumerate(LABELS)))


def save_analysis(out, prefix, r, d, z):
    distance_name = "primary_raw_correlation_distance" if prefix == "primary_raw" else prefix + "_distance"
    for name, value in ((prefix + "_pearson", r), (distance_name, d)):
        matrix_csv(out, name, value)
        np.save(out / f"{name}.npy", value)
    np.save(out / f"{prefix}_linkage.npy", z)
    write_csv(out / f"{prefix}_linkage.csv", ({"merge": i, "left": int(row[0]), "right": int(row[1]),
              "merge_distance": float(row[2]), "leaf_count": int(row[3])} for i, row in enumerate(z)))


def savefig(out, name, fig):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tree(out, name, z, support=None):
    fig, ax = plt.subplots(figsize=(15, 6))
    drawing = hierarchy.dendrogram(z, labels=LABELS, leaf_rotation=90, leaf_font_size=10, ax=ax, color_threshold=0)
    ax.set_ylabel("Algorithmic merge height")
    ax.set_title(name.replace("_", " "))
    if support is not None:
        xpos = {leaf: 5 + 10 * i for i, leaf in enumerate(drawing["leaves"])}
        xs = dict(xpos)
        sets = clades(z)
        for i, row in enumerate(z):
            x = (xs[int(row[0])] + xs[int(row[1])]) / 2
            xs[27 + i] = x
            if support[sets[i]] >= .70 and len(sets[i]) < 27:
                ax.annotate(f"{100 * support[sets[i]]:.0f}%", (x, row[2]), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8)
        ax.text(.01, .98, "Labels: non-root clades with support >=70%; complete values in CSV", transform=ax.transAxes, va="top", fontsize=9)
    savefig(out, name, fig)


def heatmap(out, name, matrix, order, correlation=False):
    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(matrix[np.ix_(order, order)], cmap="coolwarm" if correlation else "viridis",
                      vmin=-1 if correlation else 0, vmax=1)
    ax.set_xticks(range(27), [LABELS[i] for i in order], rotation=90, fontsize=8)
    ax.set_yticks(range(27), [LABELS[i] for i in order], fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson r" if correlation else "Coassignment fraction")
    ax.set_title(name.replace("_", " "))
    savefig(out, name, fig)


def bootstrap(out, x, z):
    config = {"B": B, "seed": SEED, "rng": "numpy.random.default_rng / PCG64", "unit": "TRAIN comments",
              "rows_per_replicate": len(x), "replacement": True, "linkage": "ward", "optimal_ordering": True,
              "sampling_population": "raw_rater_average_matrix.npy rows in train_matrix_row_ids.csv order"}
    write_json(out / "bootstrap_config.json", config)
    rng = np.random.default_rng(SEED)
    linkages, provenance = [], []
    for b in range(B):
        indices = rng.integers(0, len(x), size=len(x), dtype=np.int64)
        require(indices.min() >= 0 and indices.max() < len(x), "Bootstrap index out of TRAIN range")
        _, d = correlations(x[indices])
        linkages.append(linkage_from(d))
        provenance.append({"replicate": b, "sample_count": len(indices), "min_index": int(indices.min()),
                           "max_index": int(indices.max()), "sample_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()})
        if (b + 1) % 10 == 0:
            print(f"BOOTSTRAP {b + 1}/{B}", flush=True)
    linkages = np.stack(linkages)
    np.save(out / "primary_bootstrap_linkages.npy", linkages)
    write_csv(out / "bootstrap_sampling_audit.csv", provenance)
    counts = Counter(c for replicate in linkages for c in clades(replicate))
    support = {c: counts[c] / B for c in clades(z)}
    write_csv(out / "primary_clade_bootstrap_support.csv", ({"node_id": f"raw_cluster_{i + 1:03d}",
              "leaf_count": len(c), "leaf_labels": "|".join(sorted(c)), "occurrences": counts[c],
              "bootstrap_B": B, "bootstrap_support": support[c], "stable_70": support[c] >= .70,
              "trivial_root_clade": len(c) == 27} for i, c in enumerate(clades(z))))
    return linkages, support


def flat_statistics(out, d, z, bootstrap_z):
    candidates, stability, replicate_rows, partitions, coassignments = [], [], [], {}, {}
    for k in KS:
        full = cut(z, k)
        partitions[k] = full
        sizes = np.bincount(full)
        candidates.append({"k": k, "actual_k": len(sizes), "cluster_sizes": "|".join(map(str, sorted(sizes))),
                           "singletons": int((sizes == 1).sum()), "min_size": int(sizes.min()), "max_size": int(sizes.max()),
                           "silhouette": float(metrics.silhouette_score(d, full, metric="precomputed"))})
        write_csv(out / f"primary_clusters_k{k}.csv", ({"emotion": e, "cluster": int(full[i])} for i, e in enumerate(LABELS)))
        agreements = []
        co = np.zeros((27, 27), dtype=np.float64)
        for b, bz in enumerate(bootstrap_z):
            part = cut(bz, k)
            ari = float(metrics.adjusted_rand_score(full, part))
            nmi = float(metrics.normalized_mutual_info_score(full, part, average_method="arithmetic"))
            agreements.append((ari, nmi))
            replicate_rows.append({"replicate": b, "k": k, "ARI": ari, "NMI": nmi})
            co += part[:, None] == part[None, :]
        a = np.array(agreements)
        row = {"k": k}
        for j, key in enumerate(("ARI", "NMI")):
            row.update({key + "_mean": float(a[:, j].mean()), key + "_sd": float(a[:, j].std(ddof=1)),
                        key + "_min": float(a[:, j].min()), key + "_max": float(a[:, j].max())})
        stability.append(row)
        co /= B
        matrix_csv(out, f"primary_bootstrap_coassignment_k{k}", co)
        coassignments[k] = co
    write_csv(out / "primary_flat_cluster_candidates.csv", candidates)
    write_csv(out / "primary_bootstrap_stability_by_k.csv", stability)
    write_csv(out / "primary_bootstrap_partition_metrics.csv", replicate_rows)
    return candidates, stability, partitions, coassignments


def common_clade(z, a, b):
    target = {a, b}
    for i, c in enumerate(clades(z)):
        if target <= c:
            return c, float(z[i, 2])
    raise RuntimeError("Missing common clade")


def posthoc_pairs(out, name, pairs, r, d, z, rs, ds, zs, support):
    rows = []
    for a, b in pairs:
        i, j = LABELS.index(a), LABELS.index(b)
        c, height = common_clade(z, a, b)
        cs, heights = common_clade(zs, a, b)
        rows.append({"emotion_a": a, "emotion_b": b, "primary_raw_correlation": float(r[i, j]),
                     "primary_distance": float(d[i, j]), "primary_merge_height": height,
                     "primary_smallest_common_clade": "|".join(sorted(c)), "primary_clade_size": len(c),
                     "primary_bootstrap_support": support[c], "secondary_correlation": float(rs[i, j]),
                     "secondary_distance": float(ds[i, j]), "secondary_merge_height": heights,
                     "secondary_clade_size": len(cs), "secondary_smallest_common_clade": "|".join(sorted(cs))})
    write_csv(out / name, rows)
    return rows


def compare_trees(out, name, z, other, d, other_d):
    c1, v1 = hierarchy.cophenet(z, distance.squareform(d))
    c2, v2 = hierarchy.cophenet(other, distance.squareform(other_d))
    quality = {"first_cophenetic_correlation": float(c1), "second_cophenetic_correlation": float(c2),
               "cophenetic_vector_similarity": float(np.corrcoef(v1, v2)[0, 1]),
               "shared_nontrivial_clades": len((set(clades(z)) & set(clades(other))) - {frozenset(LABELS)})}
    require(all(np.isfinite(v) for k, v in quality.items()), "Invalid cophenetic statistics")
    rows = [{"k": k, "ARI": float(metrics.adjusted_rand_score(cut(z, k), cut(other, k))),
             "NMI": float(metrics.normalized_mutual_info_score(cut(z, k), cut(other, k), average_method="arithmetic"))} for k in KS]
    write_csv(out / name, rows)
    return quality, rows


def render_remaining(out, x, y, rater_stats, support, candidates, stability, comparison, r, rs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(range(27), x.sum(axis=0), label="Raw rating mass")
    axes[0].plot(range(27), y.sum(axis=0), "o", label="Simplified positive comments")
    axes[0].set_xticks(range(27), LABELS, rotation=90)
    axes[0].set_ylabel("Training support (different definitions)")
    axes[0].legend()
    distribution = rater_stats["distribution"]
    axes[1].bar(list(map(str, distribution)), list(distribution.values()))
    axes[1].set_xlabel("Unique raters per TRAIN comment (all rows)")
    axes[1].set_ylabel("Comments")
    savefig(out, "raw_label_rater_statistics", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in ("ARI", "NMI"):
        ax.errorbar(list(KS), [row[key + "_mean"] for row in stability], yerr=[row[key + "_sd"] for row in stability], marker="o", label=key)
    ax.set_xlabel("k"); ax.set_ylabel("Bootstrap agreement: mean +/- SD"); ax.legend()
    savefig(out, "primary_bootstrap_stability_by_k", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in candidates:
        sizes = list(map(int, row["cluster_sizes"].split("|")))
        ax.scatter([row["k"]] * len(sizes), sizes, alpha=.7)
    ax.set_xlabel("k"); ax.set_ylabel("Cluster size (27 emotions total)")
    savefig(out, "primary_cluster_size_by_k", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist([v for c, v in support.items() if len(c) < 27], bins=np.linspace(0, 1, 11))
    ax.axvline(.70, color="red"); ax.set_xlabel("Exact-clade bootstrap support (excluding trivial root)"); ax.set_ylabel("Clades")
    savefig(out, "bootstrap_support_distribution", fig)
    tri = np.triu_indices(27, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r[tri], rs[tri], s=12)
    ax.set_xlabel("PRIMARY raw-rater Pearson r"); ax.set_ylabel("SECONDARY simplified Pearson r")
    savefig(out, "raw_vs_simplified_pair_correlation_scatter", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in ("ARI", "NMI"):
        ax.plot(list(KS), [row[key] for row in comparison], "o-", label=key)
    ax.set_xlabel("k"); ax.set_ylabel("Raw vs simplified partition agreement"); ax.legend()
    savefig(out, "raw_vs_simplified_flat_partition_similarity_by_k", fig)


def reports(out, identity, alignment, support, candidates, stability, pairs, quality, paper_pairs, error_pairs, zs, z, sensitivity_quality):
    stable = sorted(((c, s) for c, s in support.items() if len(c) < 27 and s >= .70), key=lambda item: (-item[1], len(item[0])))
    weak = sorted(((c, s) for c, s in support.items() if len(c) < 27 and s < .70), key=lambda item: item[1])
    selection = "NO_UNIQUE_K_SUPPORTED"
    usable = [c for c, s in zip(candidates, stability) if s["ARI_mean"] >= .80 and s["NMI_mean"] >= .80 and c["singletons"] == 0 and c["max_size"] <= 20]
    selected_k = None
    eligible = [(c, s) for c, s in zip(candidates, stability) if c["singletons"] == 0 and c["max_size"] <= 20]
    if len(eligible) >= 2:
        by_silhouette = sorted(eligible, key=lambda item: item[0]["silhouette"], reverse=True)
        by_stability = sorted(eligible, key=lambda item: item[1]["ARI_mean"], reverse=True)
        winner = by_silhouette[0][0]
        if (winner["k"] in [c["k"] for c in usable] and winner["k"] == by_stability[0][0]["k"]
                and winner["silhouette"] - by_silhouette[1][0]["silhouette"] >= .02
                and by_stability[0][1]["ARI_mean"] - by_stability[1][1]["ARI_mean"] >= .05):
            selection = "CLEAR_CANDIDATE_K"
            selected_k = winner["k"]
    readiness = "READY_WITH_CAUTION" if usable else "NOT_STABLE_ENOUGH"
    recommended = "hierarchy_primary_support70.json" if usable else None
    decision = {"selection": selection, "selected_k": selected_k, "readiness": readiness, "recommended_artifact": recommended,
                "stable_balanced_candidate_k": [row["k"] for row in usable],
                "rule": "Predeclared heuristics: readiness needs a k with mean ARI>=0.80, NMI>=0.80, no singleton, largest cluster<=20/27. Clear k requires the same eligible winner in silhouette and mean ARI, margins>=0.02 and >=0.05 respectively, and at least two eligible candidates. Full binary tree retained.",
                "exp4b_launched": False, "human_review_needed": True}
    write_json(out / "EXP4B_READINESS.json", decision)
    table = "|k|silhouette|sizes|singletons|ARI mean|ARI SD|NMI mean|\n|---|---|---|---|---|---|---|\n"
    table += "\n".join(f"|{c['k']}|{c['silhouette']:.4f}|{c['cluster_sizes']}|{c['singletons']}|{s['ARI_mean']:.4f}|{s['ARI_sd']:.4f}|{s['NMI_mean']:.4f}|" for c, s in zip(candidates, stability))
    (out / "PRIMARY_HIERARCHY_CANDIDATE_SELECTION.md").write_text(f"# Candidate selection\n\n{selection}\n\n{table}\n\n{decision['rule']}\n\nFull tree remains PRIMARY; exact-clade support must be inspected. These heuristic thresholds are design choices, not established statistical tests. Readiness is provisional: {readiness}.\n", encoding="utf-8")
    changed = sorted(pairs, key=lambda p: p["primary_raw_correlation"] - p["secondary_simplified_correlation"], reverse=True)
    raw_sets, secondary_sets = set(clades(z)), set(clades(zs))
    comparison_text = ("# Raw versus simplified interpretation\n\n"
        "Per-comment averages retain fractional annotator agreement; simplified labels record consensus-selected binary outcomes. Neither implies higher classification F1.\n\n"
        "Largest increases in raw Pearson correlation:\n" + "\n".join(f"- {p['emotion_a']} / {p['emotion_b']}: delta={p['primary_raw_correlation']-p['secondary_simplified_correlation']:.4f}" for p in changed[:10]) +
        "\n\nLargest decreases:\n" + "\n".join(f"- {p['emotion_a']} / {p['emotion_b']}: delta={p['primary_raw_correlation']-p['secondary_simplified_correlation']:.4f}" for p in changed[-10:][::-1]) +
        "\n\nShared non-root clades:\n" + "\n".join("- " + ", ".join(sorted(c)) for c in sorted((raw_sets & secondary_sets) - {frozenset(LABELS)}, key=lambda c: (len(c), sorted(c)))) +
        "\n\nRaw-only clades:\n" + "\n".join("- " + ", ".join(sorted(c)) for c in sorted(raw_sets - secondary_sets, key=lambda c: (len(c), sorted(c)))) +
        "\n\nRare-label positions are documented in rare_label_tree_positions.csv (lowest TRAIN simplified-support quartile, ties included).\n"
        "PRIMARY alone was bootstrapped as specified; no comparative claim that raw is more stable is supported. Fractional information is lost by simplification, but its downstream usefulness remains untested.\n"
        f"\nAll-zero-row sensitivity: {json.dumps(sensitivity_quality)}. All-row PRIMARY remains frozen.\n")
    (out / "RAW_VS_SIMPLIFIED_INTERPRETATION.md").write_text(comparison_text, encoding="utf-8")
    closest = sorted(pairs, key=lambda p: p["primary_distance"])[:20]
    distant = sorted(pairs, key=lambda p: p["primary_distance"], reverse=True)[:20]
    write_csv(out / "top20_primary_closest_pairs.csv", closest)
    write_csv(out / "top20_primary_most_distant_pairs.csv", distant)
    outline = hierarchy.dendrogram(z, labels=LABELS, no_plot=True)["ivl"]
    lines = ["# Exp4A result", "", "1. Exp2 sentiment v0 supplies broad groups; Exp4A studies finer TRAIN annotation relations.",
             "2. PRIMARY averages all available rater rows per TRAIN comment; SECONDARY uses the same TRAIN simplified multi-hot labels.",
             f"3. Alignment: {alignment['matched_train_ids']}/43410; raw rows: {alignment['raw_train_rows']}; evaluation-ID intersections: zero.",
             "4. Canonical IDs and texts match raw records. dev/test labels are not parsed. neutral ID 27 is excluded from both matrices.",
             "5. Averaging retains fractional agreement; unclear/no-label rows are retained by the supplied specification and separately sensitivity-tested.",
             "6. Pearson correlates 27 emotion vectors across 43410 comments; distance is 1-r. Ward receives the condensed vector with optimal_ordering=True.",
             "7. Ward on correlation distance follows official code; its heights do not have the general Euclidean minimum-variance interpretation.",
             "8. PRIMARY algorithmic leaf order: " + " -> ".join(outline), "", "Closest TRAIN raw pairs:"]
    lines += [f"- {p['emotion_a']} / {p['emotion_b']}: r={p['primary_raw_correlation']:.4f}, distance={p['primary_distance']:.4f}" for p in closest[:10]]
    lines += ["", "Stable non-root clades (>=70%):"] + [f"- {', '.join(sorted(c))}: {s:.1%}" for c, s in stable]
    lines += ["", "Weak non-root clades (<70%):"] + [f"- {', '.join(sorted(c))}: {s:.1%}" for c, s in weak]
    lines += ["", table, "", f"Flat-k decision: {selection}; provisional readiness: {readiness}.",
              f"Exp4B candidate: {recommended or 'None pending scientific review'}. Full branches are retained; support70 does not prune or rewire.",
              "The full tree is structurally deeper than v0; semantic usefulness and classification gains are not established.",
              "v0 ARI/NMI and sentiment purity/composition are in primary_vs_sentiment_v0.csv; disagreement is descriptive, not failure.",
              "Raw/simplified comparison, average-linkage robustness and all-zero sensitivity are in hierarchy_quality.json.",
              json.dumps(quality, indent=2), "", "Post-hoc paper and prior error pairs:"]
    lines += [f"- {p['emotion_a']} / {p['emotion_b']}: raw r={p['primary_raw_correlation']:.4f}; simplified r={p['secondary_correlation']:.4f}; common clade size={p['primary_clade_size']}; support={p['primary_bootstrap_support']:.1%}" for p in paper_pairs + error_pairs]
    lines += ["", "These pairs were inspected only after the trees and bootstrap were constructed; proximity does not establish a cause of classifier confusion.",
              "Bootstrap support measures exact leaf-set recurrence under TRAIN-comment resampling, not the probability a semantic relation is true. Root support is trivially 100%.",
              "PRIMARY was bootstrapped B=200, seed=2026. SECONDARY bootstrap was not requested, so relative stability is not established.",
              "This is train-only paper-style analysis, not exact reproduction of full-data Figure 2. No classifier, LGSA modification, Exp4B or HPC job was launched.",
              "Candidate-k and readiness criteria are conservative heuristics declared before execution in config.json. Original raw download date is unknown; audit timestamps are recorded."]
    (out / "EXP4A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    chinese = ["# Exp4A 实验总结", "", "Exp2 的旧层次只提供 Positive、Negative、Ambiguous 等粗粒度分组。本实验从训练集人工标注关系中恢复更细的结构。",
               f"训练集有 43,410 个唯一 ID，全部匹配成功，共使用 {alignment['raw_train_rows']} 条标注；与验证集、测试集的 ID 交集为零。程序只读取验证／测试 ID，不解析标签。",
               "主分析按 comment ID 汇总所有标注者的 0/1 评分再求平均，得到 43410×27 的连续评分矩阵。简化标签对照使用同一批训练样本的多标签 0/1 矩阵。neutral 单独挂在 ROOT 下。",
               "平均评分保留标注者一致程度，但不保证更高分类性能。主分析按规范保留 unclear／无标签行；另外按官方无标签行过滤规则完成训练集敏感性分析。",
               "每种情绪在训练样本上的评分向量之间计算 Pearson 相关，距离取 1-r。将距离压缩为一维向量后执行 Ward，并使用官方代码的 optimal_ordering=True；没有手工调整叶序。",
               "Ward 的严格最小方差解释需要欧氏距离。本实验直接使用相关距离，遵循官方实现，同时用 average linkage 检查稳健性。", "", "最接近的情绪对："]
    chinese += [f"- {p['emotion_a']} / {p['emotion_b']}：r={p['primary_raw_correlation']:.4f}" for p in closest[:10]]
    chinese += ["", "稳定分支（支持率至少 70%，排除必然稳定的全体情绪根节点）："] + [f"- {', '.join(sorted(c))}：{s:.1%}" for c, s in stable]
    chinese += ["", "不稳定分支："] + [f"- {', '.join(sorted(c))}：{s:.1%}" for c, s in weak]
    chinese += ["", "200 次 bootstrap 按训练 comment 整行重采样，seed=2026；每次重新计算相关、距离和树。", table,
                f"平坦聚类结论：{selection}。Exp4B 暂定状态：{readiness}。",
                f"建议候选文件：{recommended or '暂不指定，需科学复核'}。完整二叉树作为主结果，弱分支不删除、不重连。",
                "新树在结构上比旧 sentiment v0 更深，但语义意义和分类收益仍需后续检验。与旧分组的 ARI/NMI、纯度和组成见 primary_vs_sentiment_v0.csv。",
                "主树／简化树的相关差异、共享及独有分支见 RAW_VS_SIMPLIFIED_INTERPRETATION.md；稀有标签位置见 rare_label_tree_positions.csv。",
                "旧错误分析情绪对与论文常见情绪对仅在构树完成后检查，相关、公共分支大小和支持率见 paper_pair_sanity_check.csv、error_pair_tree_alignment.csv。接近并不证明分类错误的成因。",
                "没有对简化树做 bootstrap，因此不能声称平均评分得到的树更稳定；连续评分保留的信息是否有助于分类尚未验证。",
                "本实验是仅训练集的 paper-style 分析，并非完整数据 Figure 2 的精确复现。Exp4B 未启动，层次不得根据未来 F1 反向修改。"]
    (out / "NOTION_EXP4A_SUMMARY.md").write_text("\n\n".join(chinese) + "\n", encoding="utf-8")
    return decision


def accept(out, x, y, r, rs, z, zs, bootstrap_z, support, write=True):
    checks = {}
    def check(name, passed):
        checks[name] = bool(passed)
        require(passed, "Acceptance failure: " + name)
    alignment = read_json(out / "train_raw_alignment_report.json")
    leakage = read_json(out / "split_leakage_audit.json")
    ids = read_csv(out / "train_matrix_row_ids.csv")
    check("A_B_train_unique_valid_ids", len(ids) == 43410 and len({row["comment_id"] for row in ids}) == 43410 and all(row["comment_id"] for row in ids))
    check("C_train_alignment_100_percent", alignment["coverage"] == 1 and alignment["missing_ids"] == [])
    check("D_E_F_zero_evaluation_leakage", leakage["used_raw_ids_subset_train"] and leakage["used_raw_ids_intersection_dev"] == leakage["used_raw_ids_intersection_test"] == 0)
    check("G_H_I_27_emotions_neutral_excluded", x.shape == y.shape == (43410, 27) and read_json(out / "data_identity.json")["emotion_column_order"] == LABELS)
    check("J_primary_values_0_1", np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1)
    check("secondary_binary_multihot", np.isin(y, [0, 1]).all())
    for name, value in (("primary", r), ("secondary", rs)):
        check(name + "_K_L_M_N_correlation", value.shape == (27, 27) and np.isfinite(value).all() and np.allclose(value, value.T) and np.allclose(np.diag(value), 1))
    check("correlations_recomputed_from_saved_matrices", np.allclose(correlations(x)[0], r, atol=1e-12) and np.allclose(correlations(y)[0], rs, atol=1e-12))
    check("linkages_recomputed_from_saved_correlations", np.allclose(linkage_from(1-r), z, atol=1e-12) and np.allclose(linkage_from(1-rs), zs, atol=1e-12))
    check("saved_distance_files_match_correlations", np.allclose(np.load(out / "primary_raw_correlation_distance.npy"), 1-r) and np.allclose(np.load(out / "secondary_simplified_distance.npy"), 1-rs))
    check("R_bootstrap_count_200", bootstrap_z.shape == (200, 26, 4))
    sample_rows = read_csv(out / "bootstrap_sampling_audit.csv")
    require(len(sample_rows) == B, "Missing bootstrap audit rows")
    rng = np.random.default_rng(SEED)
    for b, row in enumerate(sample_rows):
        indices = rng.integers(0, 43410, size=43410, dtype=np.int64)
        require(int(row["replicate"]) == b and int(row["sample_count"]) == 43410, "Bootstrap provenance mismatch")
        require(row["sample_indices_sha256"] == hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(), "Bootstrap sample checksum mismatch")
        require(0 <= int(row["min_index"]) <= int(row["max_index"]) < 43410, "Bootstrap non-TRAIN range")
        require(hierarchy.is_valid_linkage(bootstrap_z[b]), "Invalid bootstrap tree")
    check("S_all_bootstraps_train_only_reproducible_indices", True)
    recurrence = Counter(c for bz in bootstrap_z for c in clades(bz))
    check("exact_clade_support_recomputed", all(support[c] == recurrence[c] / B for c in clades(z)))
    for name in ("hierarchy_primary_raw_full.json", "hierarchy_primary_support70.json", "hierarchy_secondary_simplified_full.json"):
        validate_tree(read_json(out / name))
    check("saved_tree_support_and_structure_match_linkage", read_json(out / "hierarchy_primary_raw_full.json") == tree(z, "raw", support)
          and read_json(out / "hierarchy_primary_support70.json") == tree(z, "raw", support)
          and read_json(out / "hierarchy_secondary_simplified_full.json") == tree(zs, "simplified"))
    check("saved_matrices_match_analysis", np.array_equal(np.load(out / "raw_rater_average_matrix.npy"), x)
          and np.array_equal(np.load(out / "secondary_simplified_matrix.npy"), y))
    check("O_P_Q_T_json_reload_and_exact_leaf_coverage", True)
    check("primary_secondary_linkage_valid", hierarchy.is_valid_linkage(z) and hierarchy.is_valid_linkage(zs))
    for file in out.glob("*.json"):
        read_json(file)
    if write:
        write_json(out / "ACCEPTANCE_CHECKS.json", {"all_passed": True, "checks": checks})
    return checks


def execute(root, out, audit_only):
    versions = load_science()
    require(socket.gethostname() == "D3020-A3", "Formal/audit execution requires Ubuntu host D3020-A3")
    environment = {"hostname": socket.gethostname(), "project_root": str(root), "python_executable": sys.executable,
                   "python_version": platform.python_version(), "versions": versions, "platform": platform.platform(),
                   "git_commit": git(root, "rev-parse", "HEAD"), "git_branch": git(root, "branch", "--show-current"),
                   "git_status_before": git(root, "status", "--short"), "started_at": timestamp()}
    write_json(out / "environment.json", environment)
    protocol = {"primary": "TRAIN_ONLY_RAW_RATER_AVERAGED", "secondary": "TRAIN_ONLY_SIMPLIFIED_BINARY",
                "neutral_id": 27, "emotion_order": LABELS, "bootstrap_B": B, "bootstrap_seed": SEED,
                "distance": "1 - Pearson correlation", "linkage": "ward", "optimal_ordering": True,
                "distance_representation": "scipy.spatial.distance.squareform(D), 1D condensed vector",
                "primary_zero_label_rows": "retain all raw rows per supplied specification",
                "sensitivity_zero_label_rows": "exclude rows with sum across all 28 labels equal to zero",
                "flat_cut": "scipy.cluster.hierarchy.cut_tree(n_clusters=[k]), exact k",
                "k_values": list(KS), "exact_clade_stability_threshold": .70,
                "readiness_heuristic": "at least one k: mean bootstrap ARI>=0.80, NMI>=0.80, no singleton, max cluster<=20; otherwise NOT_STABLE_ENOUGH",
                "unique_k_policy": "same singleton-free balanced winner in silhouette and mean ARI; margins>=0.02 and >=0.05, at least two eligible candidates; else NO_UNIQUE_K_SUPPORTED",
                "reference_full_raw_tree": "not constructed", "classifier_training": False,
                "nmi_average_method": "arithmetic", "stability_sd_ddof": 1}
    write_json(out / "config.json", protocol)
    method = Path(__file__).with_name("PAPER_METHOD_AUDIT.md")
    require(method.is_file(), "Missing paper audit")
    (out / "PAPER_METHOD_AUDIT.md").write_bytes(method.read_bytes())
    write_json(out / "source_code_identity.json", {p.name: sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()})
    print("STAGE: strict data audit", flush=True)
    sums, counts, labeled_counts, annotations, identity, alignment, leakage, mapping = audit(root, out)
    if audit_only:
        write_json(out / "AUDIT_PASS.json", {"status": "EXP4A_DATA_AUDIT_PASS", "raw_train_rows": int(counts.sum()), "train_ids": 43410})
        return {"mode": "audit_only", "status": "EXP4A_DATA_AUDIT_PASS"}
    print("STAGE: PRIMARY averaged judgments and Ward hierarchy", flush=True)
    x = sums[:, :27] / counts[:, None]
    np.save(out / "raw_rater_average_matrix.npy", x)
    r, d = correlations(x)
    z = linkage_from(d)
    # Validate numerical equivalence to the official pdist construction.
    require(np.allclose(distance.pdist(x.T, metric="correlation"), distance.squareform(d), atol=1e-12), "Official pdist equivalence failed")
    save_analysis(out, "primary_raw", r, d, z)
    write_json(out / "hierarchy_primary_raw_full.json", tree(z, "raw"))
    plot_tree(out, "primary_raw_dendrogram", z)
    heatmap(out, "primary_raw_correlation_heatmap", r, hierarchy.leaves_list(z), True)
    print("STAGE: 200 TRAIN-comment bootstraps", flush=True)
    bootstrap_z, support = bootstrap(out, x, z)
    annotated = tree(z, "raw", support)
    write_json(out / "hierarchy_primary_raw_full.json", annotated)
    write_json(out / "hierarchy_primary_support70.json", copy.deepcopy(annotated))
    plot_tree(out, "primary_raw_dendrogram_bootstrap", z, support)
    candidates, stability, partitions, coassignments = flat_statistics(out, d, z, bootstrap_z)
    for k in (3, 4, 5, 6):
        heatmap(out, f"primary_bootstrap_coassignment_k{k}", coassignments[k], hierarchy.leaves_list(z))
    print("STAGE: SECONDARY simplified binary hierarchy", flush=True)
    y = np.zeros((43410, 27), dtype=np.int64)
    for i, labels in enumerate(annotations):
        for label in labels:
            if label != 27:
                y[i, label] = 1
    np.save(out / "secondary_simplified_matrix.npy", y)
    rs, ds = correlations(y)
    zs = linkage_from(ds)
    save_analysis(out, "secondary_simplified", rs, ds, zs)
    write_json(out / "hierarchy_secondary_simplified_full.json", tree(zs, "simplified"))
    plot_tree(out, "secondary_simplified_dendrogram", zs)
    heatmap(out, "secondary_simplified_correlation_heatmap", rs, hierarchy.leaves_list(zs), True)
    print("STAGE: frozen-tree comparisons and post-hoc analyses", flush=True)
    quality, comparison = compare_trees(out, "primary_vs_secondary_tree_comparison.csv", z, zs, d, ds)
    tri = np.triu_indices(27, 1)
    quality["pairwise_correlation_similarity"] = float(np.corrcoef(r[tri], rs[tri])[0, 1])
    quality["distance_matrix_similarity"] = float(np.corrcoef(d[tri], ds[tri])[0, 1])
    quality["distance_similarity_note"] = "Affine relation makes this equal to pairwise correlation similarity"
    cooccurrence = y.T @ y
    pairs = []
    for i, j in combinations(range(27), 2):
        pairs.append({"emotion_a": LABELS[i], "emotion_b": LABELS[j], "primary_raw_correlation": float(r[i, j]),
                      "primary_distance": float(d[i, j]), "secondary_simplified_correlation": float(rs[i, j]),
                      "secondary_distance": float(ds[i, j]), "train_simplified_support_a": int(y[:, i].sum()),
                      "train_simplified_support_b": int(y[:, j].sum()), "train_raw_rating_mass_a": float(x[:, i].sum()),
                      "train_raw_rating_mass_b": float(x[:, j].sum()), "binary_gt_cooccurrence_count": int(cooccurrence[i, j])})
    write_csv(out / "emotion_pair_relationships.csv", pairs)
    write_csv(out / "primary_vs_secondary_pairwise.csv", pairs)
    paper_pairs = posthoc_pairs(out, "paper_pair_sanity_check.csv", [("anger", "annoyance"), ("fear", "nervousness"), ("joy", "excitement"), ("grief", "sadness")], r, d, z, rs, ds, zs, support)
    error_pairs = posthoc_pairs(out, "error_pair_tree_alignment.csv", [("admiration", "approval"), ("anger", "annoyance"), ("curiosity", "confusion"), ("gratitude", "admiration"), ("love", "admiration"), ("amusement", "joy")], r, d, z, rs, ds, zs, support)
    sentiment_labels = np.array([{"Positive": 0, "Negative": 1, "Ambiguous": 2}[mapping[e]] for e in LABELS])
    sentiment_rows = []
    for k, part in partitions.items():
        composition = {str(c): dict(Counter(mapping[LABELS[i]] for i in range(27) if part[i] == c)) for c in sorted(set(part))}
        purity = sum(max(counts.values()) for counts in composition.values()) / 27
        sentiment_rows.append({"k": k, "ARI": float(metrics.adjusted_rand_score(sentiment_labels, part)),
                               "NMI": float(metrics.normalized_mutual_info_score(sentiment_labels, part, average_method="arithmetic")),
                               "purity": purity, "sentiment_composition": json.dumps(composition),
                               "note": "Descriptive comparison; purity tends to increase with k"})
    write_csv(out / "primary_vs_sentiment_v0.csv", sentiment_rows)
    za = linkage_from(d, "average")
    np.save(out / "primary_average_linkage.npy", za)
    plot_tree(out, "primary_average_linkage_dendrogram", za)
    average_quality, _ = compare_trees(out, "primary_ward_vs_average_comparison.csv", z, za, d, d)
    xs = sums[:, :27] / labeled_counts[:, None]
    rr, dd = correlations(xs)
    zz = linkage_from(dd)
    save_analysis(out, "sensitivity_official_row_filter", rr, dd, zz)
    plot_tree(out, "sensitivity_official_row_filter_dendrogram", zz)
    sensitivity_quality, _ = compare_trees(out, "primary_vs_official_row_filter_comparison.csv", z, zz, d, dd)
    sensitivity_quality["same_train_ids"] = True
    sensitivity_quality["excluded_no_label_rows"] = int(counts.sum() - labeled_counts.sum())
    quality = {"primary_raw": quality["first_cophenetic_correlation"], "secondary_simplified": quality["second_cophenetic_correlation"],
               "raw_vs_simplified": quality, "ward_vs_average": average_quality, "official_row_filter_sensitivity": sensitivity_quality}
    write_json(out / "hierarchy_quality.json", quality)
    rare_rows = []
    train_support = y.sum(axis=0)
    threshold = float(np.quantile(train_support, .25))
    for i, label in enumerate(LABELS):
        if train_support[i] <= threshold:
            smallest = next(c for c in clades(z) if label in c)
            secondary = next(c for c in clades(zs) if label in c)
            rare_rows.append({"emotion": label, "simplified_train_support": int(train_support[i]), "rare_threshold_q25": threshold,
                              "primary_first_clade": "|".join(sorted(smallest)), "primary_clade_support": support[smallest],
                              "secondary_first_clade": "|".join(sorted(secondary))})
    write_csv(out / "rare_label_tree_positions.csv", rare_rows)
    render_remaining(out, x, y, read_json(out / "rater_count_statistics.json"), support, candidates, stability, comparison, r, rs)
    print("STAGE: acceptance checks and reports", flush=True)
    accept(out, x, y, r, rs, z, zs, bootstrap_z, support)
    decision = reports(out, identity, alignment, support, candidates, stability, pairs, quality, paper_pairs, error_pairs, zs, z, sensitivity_quality)
    required = ["DATA_AUDIT.md", "PAPER_METHOD_AUDIT.md", "data_identity.json", "train_raw_alignment.csv", "train_raw_alignment_report.json",
                "split_leakage_audit.json", "rater_count_statistics.json", "rater_count_distribution.csv", "raw_rater_average_matrix.npy",
                "primary_raw_pearson.csv", "primary_raw_pearson.npy", "primary_raw_correlation_distance.csv", "primary_raw_correlation_distance.npy",
                "primary_raw_linkage.csv", "primary_raw_linkage.npy", "hierarchy_primary_raw_full.json", "hierarchy_primary_support70.json",
                "primary_clade_bootstrap_support.csv", "bootstrap_config.json", "primary_bootstrap_stability_by_k.csv", "primary_flat_cluster_candidates.csv",
                "secondary_simplified_matrix.npy", "secondary_simplified_pearson.csv", "secondary_simplified_distance.csv", "secondary_simplified_linkage.csv",
                "hierarchy_secondary_simplified_full.json", "emotion_pair_relationships.csv", "primary_vs_secondary_pairwise.csv",
                "primary_vs_secondary_tree_comparison.csv", "paper_pair_sanity_check.csv", "error_pair_tree_alignment.csv", "primary_vs_sentiment_v0.csv",
                "hierarchy_quality.json", "environment.json", "config.json", "ACCEPTANCE_CHECKS.json", "EXP4A_REPORT.md", "NOTION_EXP4A_SUMMARY.md",
                "RAW_VS_SIMPLIFIED_INTERPRETATION.md", "PRIMARY_HIERARCHY_CANDIDATE_SELECTION.md"]
    for k in KS:
        required += [f"primary_clusters_k{k}.csv", f"primary_bootstrap_coassignment_k{k}.csv"]
    figures = ["primary_raw_correlation_heatmap", "primary_raw_dendrogram", "primary_raw_dendrogram_bootstrap", "raw_label_rater_statistics",
               "primary_bootstrap_stability_by_k", "primary_cluster_size_by_k", "bootstrap_support_distribution", "secondary_simplified_correlation_heatmap",
               "secondary_simplified_dendrogram", "raw_vs_simplified_pair_correlation_scatter", "raw_vs_simplified_flat_partition_similarity_by_k",
               "primary_average_linkage_dendrogram"] + [f"primary_bootstrap_coassignment_k{k}" for k in (3, 4, 5, 6)]
    required += [f"{figure}.{extension}" for figure in figures for extension in ("png", "pdf")]
    require(all((out / name).is_file() and (out / name).stat().st_size > 0 for name in required), "Missing mandatory artifact")
    environment["finished_at"] = timestamp()
    environment["git_status_after"] = git(root, "status", "--short")
    write_json(out / "environment.json", environment)
    write_json(out / "ARTIFACT_MANIFEST.json", {"algorithm": "sha256", "excludes": ["ARTIFACT_MANIFEST.json", "SUCCESS.json", "NOTIFICATIONS.json"],
               "files": [{"filename": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name not in {"ARTIFACT_MANIFEST.json", "SUCCESS.json", "NOTIFICATIONS.json"}]})
    verify_manifest(out)
    result = {"status": "EXP4A_COMPLETE_VERIFIED", "primary_hierarchy": "TRAIN_ONLY_RAW_RATER_AVERAGED",
              "secondary_hierarchy": "TRAIN_ONLY_SIMPLIFIED_BINARY", "bootstrap_B": B, "seed": SEED,
              "raw_train_rows": alignment["raw_train_rows"], "train_ids": 43410, "leakage": "ZERO",
              "readiness": decision["readiness"], "recommended_artifact": decision["recommended_artifact"],
              "output_dir": str(out), "completed_at": timestamp(), "artifact_manifest_sha256": sha(out / "ARTIFACT_MANIFEST.json")}
    write_json(out / "SUCCESS.json", result)
    return result


def verify_manifest(out):
    manifest = read_json(out / "ARTIFACT_MANIFEST.json")
    for item in manifest["files"]:
        path = out / item["filename"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], f"Artifact hash mismatch: {path}")
    return len(manifest["files"])


def verify_saved(out):
    load_science()
    result = read_json(out / "SUCCESS.json")
    require(result["status"] == "EXP4A_COMPLETE_VERIFIED", "Missing valid success status")
    require(result["artifact_manifest_sha256"] == sha(out / "ARTIFACT_MANIFEST.json"), "Manifest identity mismatch")
    n = verify_manifest(out)
    for name in ("hierarchy_primary_raw_full.json", "hierarchy_primary_support70.json", "hierarchy_secondary_simplified_full.json"):
        validate_tree(read_json(out / name))
    support_rows = read_csv(out / "primary_clade_bootstrap_support.csv")
    support = {frozenset(row["leaf_labels"].split("|")): float(row["bootstrap_support"]) for row in support_rows}
    accept(out, np.load(out / "raw_rater_average_matrix.npy"), np.load(out / "secondary_simplified_matrix.npy"),
           np.load(out / "primary_raw_pearson.npy"), np.load(out / "secondary_simplified_pearson.npy"),
           np.load(out / "primary_raw_linkage.npy"), np.load(out / "secondary_simplified_linkage.npy"),
           np.load(out / "primary_bootstrap_linkages.npy"), support, write=False)
    verify_manifest(out)
    print("SAVED_ARTIFACT_VERIFICATION: PASS", "manifest_files=", n)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (args.output or root / "outputs/exp4a_data_driven_hierarchy").resolve()
    require(not (args.audit_only and args.verify_only), "Modes conflict")
    if args.verify_only:
        verify_saved(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), f"Refusing to overwrite nonempty output directory: {out}; choose a new --output")
    notifications = {"start": notify("🚀 FYP Exp4A " + ("data audit" if args.audit_only else "formal analysis") + " started: train-only human judgments")}
    try:
        result = execute(root, out, args.audit_only)
    except BaseException as exc:
        write_json(out / "FAILURE.json", {"status": "EXP4A_FAILED", "error": repr(exc), "traceback": traceback.format_exc(), "time": timestamp()})
        notifications["failure"] = notify("❌ FYP Exp4A failed; check Ubuntu logs: " + repr(exc)[:180])
        write_json(out / "NOTIFICATIONS.json", notifications)
        raise
    notifications["success"] = notify("✅ FYP Exp4A " + ("data audit passed" if args.audit_only else "finished: raw-rater + simplified hierarchies ready; " + result["readiness"]))
    write_json(out / "NOTIFICATIONS.json", notifications)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
