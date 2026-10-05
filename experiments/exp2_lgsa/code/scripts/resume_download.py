"""Persistent HTTP Range download with SHA256 verification and atomic promotion.

Use for large archives/wheels with a known official SHA256. .part files survive
process/SSH interruption. Servers without Range support restart that file only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_sha256, attempts=8, timeout=60, backoff=2):
    destination = Path(destination)
    expected_sha256 = expected_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A verified source SHA256 is required")
    if not url.startswith(("https://", "http://")):
        raise ValueError("Only HTTP(S) download URLs are supported")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    meta = destination.with_name(destination.name + ".part.json")
    lock = destination.with_name(destination.name + ".lock")
    with lock.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            if sha256(destination) != expected_sha256:
                raise ValueError("Existing completed file has wrong SHA256; inspect it before replacement")
            print(f"VERIFIED_SKIP {destination.name}", flush=True)
            return destination
        if meta.exists():
            if json.loads(meta.read_text())["sha256"] != expected_sha256:
                raise ValueError("Partial file belongs to a different asset version; use a new destination")
        elif part.exists():
            raise ValueError("Unidentified partial file; cannot safely append")
        else:
            # Save identity before downloading any bytes; a half-written identity
            # file fails closed rather than mixing different downloads.
            meta.write_text(json.dumps({"sha256": expected_sha256}) + "\n")

        # Handles interruption after the last byte but before rename/checksum.
        if part.exists() and sha256(part) == expected_sha256:
            os.replace(part, destination)
            meta.unlink()
            return destination

        last_error = None
        for attempt in range(attempts):
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Accept-Encoding": "identity", "User-Agent": "LGSA-FYP-download/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = response.status
                    total = None
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset or int(match[2]) < offset:
                            raise ValueError("Invalid Content-Range; refusing to append")
                        total = int(match[3])
                        if int(match[2]) != total - 1:
                            raise ValueError("Unexpected partial range endpoint")
                        mode = "ab" if offset else "wb"
                        print(f"RESUME {destination.name} from byte {offset}", flush=True)
                    elif status == 200:
                        if offset:
                            print(f"NO_RANGE_SUPPORT: restarting only {destination.name}", flush=True)
                        offset = 0
                        mode = "wb"
                        if response.headers.get("Content-Length"):
                            total = int(response.headers["Content-Length"])
                    else:
                        raise ValueError(f"Unexpected HTTP status {status}")
                    with part.open(mode) as output:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            output.flush()
                        os.fsync(output.fileno())
                    if total is not None and part.stat().st_size != total:
                        raise OSError("Incomplete response; preserving .part for resumption")
                if sha256(part) != expected_sha256:
                    # Keep evidence; never promote unverified content to final.
                    raise ValueError("SHA256 mismatch: partial file retained for inspection; no file promoted")
                os.replace(part, destination)
                meta.unlink()
                print(f"VERIFIED_COMPLETE {destination.name}", flush=True)
                return destination
            except (OSError, urllib.error.URLError) as error:
                last_error = error
                print(f"Download interrupted (attempt {attempt+1}/{attempts}); partial bytes retained", flush=True)
                if attempt + 1 < attempts:
                    time.sleep(min(backoff * (attempt + 1), 10))
        raise RuntimeError(f"Retries exhausted; rerun with the same destination to resume {part.name}") from last_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    download(args.url, args.output, args.sha256, args.attempts)
