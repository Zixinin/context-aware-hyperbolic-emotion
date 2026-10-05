"""Standalone exit-code gate between smoke and full in the same allocation."""
import argparse
from pathlib import Path
from gates import smoke_gate
from prepare import verify_assets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('smoke_run', type=Path)
    args = parser.parse_args()
    smoke_gate(args.smoke_run.resolve(), verify_assets())
    print('GOEMOTIONS_SMOKE_ACCEPTANCE_PASS', flush=True)
