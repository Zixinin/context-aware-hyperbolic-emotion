"""Headless figures; descriptive diagnostics never feed model selection."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def frequency_chart(frame, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ordered = frame.sort_values("train_count")
    ax.bar(ordered["label"], ordered["train_count"], color=["#bf6248" if x else "#447caa" for x in ordered["is_rare"]])
    ax.set(ylabel="Training label occurrences", title="GoEmotions: full training split (rare labels highlighted)")
    ax.tick_params(axis="x", rotation=70)
    save(fig, path)


def training_figures(log, directory):
    directory = Path(directory)
    for metric, filename, title in (("training_loss", "training_loss.png", "Mean training BCE loss"),
                                    ("validation_macro_f1", "validation_macro_f1.png", "Validation Macro-F1 at threshold 0.30")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(log["epoch"], log[metric], marker="o")
        ax.set(xlabel="Epoch", ylabel=metric, title=title)
        ax.set_xticks(log["epoch"])
        save(fig, directory / filename)


def evaluation_figures(frame, frequencies, directory, title):
    directory = Path(directory)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(frame["label"], frame["f1"], color="#447caa")
    ax.set(ylabel="F1", ylim=(0, 1), title=title)
    ax.tick_params(axis="x", rotation=70)
    save(fig, directory / "per_label_f1.png")
    merged = frame.merge(frequencies[["label_id", "train_count", "is_rare"]], on="label_id")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(merged["train_count"], merged["f1"], c=["#bf6248" if x else "#447caa" for x in merged["is_rare"]])
    for row in merged.itertuples():
        ax.annotate(row.label, (row.train_count, row.f1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xscale="log", xlabel="Training label frequency (log scale)", ylabel="F1", ylim=(-0.03, 1.08), title=title)
    save(fig, directory / "frequency_vs_f1.png")
