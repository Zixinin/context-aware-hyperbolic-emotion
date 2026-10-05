import argparse
import json

from src.common import ROOT
from src.data import analyze, load_official_tsv, load_prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tsv", type=str, help="Explicit directory with the four verified official files")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/dataset_analysis"))
    args = parser.parse_args()
    dataset = load_official_tsv(args.official_tsv) if args.official_tsv else load_prepared()[0]
    frequencies, statistics, rare = analyze(dataset, args.output_dir)
    print(frequencies.to_string(index=False))
    print(json.dumps(statistics, indent=2))
    print("Rare labels (training only):", ", ".join(rare["label_names"]))


if __name__ == "__main__":
    main()
