#!/bin/bash
# Pinned Mendeley V2 raw assets; no extraction, preprocessing or training.
set -euo pipefail
cd /gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions
mkdir -p downloads/wos-v2
cd downloads/wos-v2

fetch_checked() {
    local name="$1" checksum="$2" file_id="$3"
    if [ -f "$name" ]; then
        printf '%s  %s\n' "$checksum" "$name" | sha256sum -c -
        return
    fi
    if [ -f "$name.part.sha256" ]; then
        [ "$(cat "$name.part.sha256")" = "$checksum" ] || {
            echo "Partial download version mismatch: $name"
            return 1
        }
    elif [ -e "$name.part" ]; then
        echo "Unidentified partial download: $name.part; inspect before continuing"
        return 1
    else
        printf '%s\n' "$checksum" > "$name.part.sha256"
    fi
    if [ -f "$name.part" ] &&
       printf '%s  %s\n' "$checksum" "$name.part" | sha256sum --status -c -; then
        echo "Partial file already complete: $name"
    else
        curl --fail --location --continue-at - \
          --retry 10 --retry-delay 5 --connect-timeout 30 \
          --output "$name.part" \
          "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/$file_id/file_downloaded"
    fi
    printf '%s  %s\n' "$checksum" "$name.part" | sha256sum -c -
    mv "$name.part" "$name"
}

fetch_checked WebOfScience.zip \
  b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a \
  4ed4e75a-a656-4e55-9cb2-f5638b4de00a
fetch_checked ReadMe.txt \
  6da4ab1e624c4ef61c46600d727770f873ff4db3c15541fb481e7af8a715c0ec \
  e0ed118e-529c-4e94-921e-9cdaeae7b7f8

/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo/bin/python - <<'PY'
import json
from pathlib import Path
import zipfile

archive = Path('WebOfScience.zip')
assert archive.stat().st_size == 60222421
assert Path('ReadMe.txt').stat().st_size == 2587
with zipfile.ZipFile(archive) as z:
    required = ['Meta-data/Data.xlsx', 'WOS46985/X.txt',
                'WOS46985/Y.txt', 'WOS46985/YL1.txt', 'WOS46985/YL2.txt']
    for name in required:
        item = z.getinfo(name)
        assert item.file_size > 0
        print('ARCHIVE_ENTRY_OK', name, item.file_size)
record = {
    'doi': '10.17632/9rw3vkcfy4.2',
    'source': 'https://data.mendeley.com/datasets/9rw3vkcfy4/2',
    'source_file_api': 'https://data.mendeley.com/public-api/datasets/9rw3vkcfy4/files?folder_id=root&version=2',
    'sha256': {
        'WebOfScience.zip': 'b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a',
        'ReadMe.txt': '6da4ab1e624c4ef61c46600d727770f873ff4db3c15541fb481e7af8a715c0ec'
    },
    'archive_path': str(archive.resolve()),
    'status': 'raw_download_verified',
    'preprocessing_passed': False,
    'gpu_smoke_passed': False
}
Path('source-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print('WOS_RAW_DOWNLOAD_VERIFIED')
PY
