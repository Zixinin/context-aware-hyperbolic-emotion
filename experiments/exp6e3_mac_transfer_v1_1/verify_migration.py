"""Portable stdlib verification on the destination Mac/new Ubuntu, no source access."""
import argparse,hashlib,json,platform
from pathlib import Path

def verify(base,manifest):
    rows=json.loads(Path(manifest).read_text(encoding='utf-8'));total=0
    for row in rows:
        relative=Path('payload')/row['alias']/row['relative']
        if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Unsafe manifest path')
        p=Path(base)/relative
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=row['size']:raise RuntimeError('Missing/wrong destination file: '+str(p))
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
        if h.hexdigest()!=row['sha256']:raise RuntimeError('SHA mismatch: '+str(p))
        total+=row['size']
    if not rows:raise RuntimeError('Empty migration manifest')
    return dict(files=len(rows),logical_bytes=total,verified_destination_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    result=verify(a.base,a.manifest)
    print('MAC_LOCAL_BACKUP_VERIFIED_PASS' if platform.system()=='Darwin' else 'MIGRATION_DESTINATION_VERIFIED_PASS')
    print(json.dumps(result,indent=2))
