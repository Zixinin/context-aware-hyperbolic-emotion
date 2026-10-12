"""Bounded nonfatal notifications; caller retains actual experiment exit status."""
import argparse
from pathlib import Path
from b4lib import notify, now, write

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--message', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = {**notify(args.message), 'message': args.message, 'at': now()}
    write(args.out, result)
    print('NTFY_DELIVERED' if result['delivered'] else 'NTFY_UNAVAILABLE', flush=True)

if __name__ == '__main__':
    main()
