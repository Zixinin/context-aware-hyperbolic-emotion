"""Bounded, nonfatal ntfy delivery; callers preserve the experiment exit code."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from exp5lib import now, write


def deliver(message):
    url = os.environ.get("FYP_NTFY_URL", "https://ntfy.sh/zixin-fyp-4090-1119")
    try:
        request = urllib.request.Request(url, data=message.encode("utf-8"))
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        return {"delivered": True, "message_id": payload.get("id"), "at": now(), "message": message}
    except Exception as exc:
        return {"delivered": False, "error_type": type(exc).__name__, "at": now(), "message": message}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = deliver(args.message)
    write(args.out, result)
    print("NTFY_DELIVERED" if result["delivered"] else "NTFY_UNAVAILABLE", flush=True)


if __name__ == "__main__":
    main()
