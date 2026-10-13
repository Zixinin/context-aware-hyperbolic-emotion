"""Bounded nonfatal notification retries; never turn delivery into run success."""
import argparse
import time
from m4lib import write
from b4lib import notify

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--message', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    receipts = []
    for attempt in range(1, 4):
        r = notify(a.message)
        receipts.append(dict(attempt=attempt, **r))
        if r['delivered']:
            break
        if attempt < 3:
            time.sleep(1)
    write(a.out, {'message': a.message, 'delivered': receipts[-1]['delivered'], 'attempts': receipts})
    print('NTFY_DELIVERED' if receipts[-1]['delivered'] else 'NTFY_UNAVAILABLE: inspect local logs; experiment state is independent', flush=True)
