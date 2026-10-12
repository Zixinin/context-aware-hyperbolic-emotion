"""Structure-only validation; safe on Mac, does NOT authorize formal training."""
import argparse
import json
from pathlib import Path
from hierarchy_adapter import load_hierarchy
from b4lib import sha, write

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hierarchy', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    h = load_hierarchy(args.hierarchy)
    result = {**h['summary'], 'hierarchy_sha256': sha(args.hierarchy)}
    if args.out:
        if args.out.exists():
            raise RuntimeError('Never overwrite a validation record')
        write(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
