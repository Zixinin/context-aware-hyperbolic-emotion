"""Pack review evidence without copying large checkpoints or modifying seals."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--failed',action='store_true');a=p.parse_args()
root=a.root.resolve();destination=root/'review';destination.mkdir(exist_ok=True)
name='Exp6C1S_Train_failure_review.tar.gz' if a.failed else 'Exp6C1S_Train_review.tar.gz'
out=destination/name
assert not out.exists(),'Refuse overwrite of earlier review export'
included={};omitted=[]
with tarfile.open(out,'x:gz') as archive:
    for file in sorted(root.rglob('*')):
        rel=file.relative_to(root)
        if not file.is_file() or {'review','runtime_cache','__pycache__'} & set(rel.parts):continue
        if file.suffix=='.pt':omitted.append(str(rel));continue
        # Snapshot active tee logs into archive bytes; never mutate sealed root.
        data=file.read_bytes();included[str(rel)]=hashlib.sha256(data).hexdigest()
        info=tarfile.TarInfo(str(rel));info.size=len(data);archive.addfile(info,io.BytesIO(data))
    meta=dict(kind='FAILURE_EVIDENCE' if a.failed else 'COMPLETED_VALIDATION_ONLY_PILOT_REVIEW',
        source_root=str(root),omitted_large_checkpoints=omitted,included_sha256=included,
        note='Full checkpoints stay on Ubuntu. Downloaded review does not claim local checkpoint reload. Runtime caches omitted.')
    data=(json.dumps(meta,indent=2,ensure_ascii=False)+'\n').encode('utf-8');info=tarfile.TarInfo('REVIEW_EXPORT.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
print('REVIEW_ARCHIVE='+str(out),flush=True)
print(hashlib.sha256(out.read_bytes()).hexdigest()+'  '+str(out),flush=True)
