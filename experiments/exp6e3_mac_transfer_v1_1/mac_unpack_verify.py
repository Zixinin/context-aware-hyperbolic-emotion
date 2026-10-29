"""Verify archive and extract regular manifest-listed files into a fresh Mac target."""
import argparse,hashlib,json,os,platform,tarfile
from pathlib import Path,PurePosixPath

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def unpack(archive,manifest,target,expected):
    archive,manifest,target=map(Path,[archive,manifest,target])
    if sha(archive)!=expected:raise RuntimeError('Archive SHA256 mismatch; nothing extracted')
    rows=json.loads(manifest.read_text(encoding='utf-8'))
    indexed={r['path']:r for r in rows}
    if not rows or len(indexed)!=len(rows):raise RuntimeError('Empty/duplicate manifest')
    for r in rows:
        p=PurePosixPath(r['path'])
        if p.is_absolute() or '..' in p.parts or str(p)!=r['path'] or not p.parts or r['size']<0:
            raise RuntimeError('Unsafe manifest path/size')
    # Existing output is never overwritten. Preserve an interrupted attempt for review.
    if target.exists() or target.is_symlink():raise RuntimeError('Extraction target exists; preserve it and use a fresh target')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(rows) or len({m.name for m in members})!=len(rows):raise RuntimeError('Archive member count/duplicates differ')
        for m in members:
            if not m.isfile() or m.name not in indexed or m.size!=indexed[m.name]['size']:
                raise RuntimeError('Unexpected archive path/type/size')
        target.mkdir(parents=True,exist_ok=False)
        for m in members:
            p=target/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,p.open('xb') as dst:
                for b in iter(lambda:src.read(4*1024**2),b''):dst.write(b)
            if sha(p)!=indexed[m.name]['sha256']:raise RuntimeError('Extracted destination SHA256 mismatch: '+m.name)
    # Re-read every final destination independently after extraction.
    for r in rows:
        p=target/r['path']
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:
            raise RuntimeError('Final target verification failed: '+r['path'])
    result=dict(files=len(rows),logical_bytes=sum(r['size'] for r in rows),destination=str(target.resolve()),
        all_destination_sha256_verified=True,archive_sha256=expected,
        status='MAC_LOCAL_BACKUP_VERIFIED_PASS' if platform.system()=='Darwin' else 'LOCAL_RESULT_PACKAGE_VERIFIED_PASS',
        scope='New Exp6E-3 run and source dependencies, not entire historical project backup')
    (target.parent/('MAC_VERIFICATION_'+target.name+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
    print(json.dumps(unpack(a.archive,a.manifest,a.target,a.sha256),indent=2))
