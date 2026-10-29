"""Seal a completed, quiescent new run for Mac transfer; no training or Test."""
import argparse,gzip,hashlib,json,os,shutil,stat,tarfile,tempfile
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
BASE=REPO/'results/exp6e3_matched_gamma_coordinate'
RECORDS=Path('/home/zixinfeng/Documents/FYP_migration_records')
MIN_MODEL_BYTES=100*1024**2
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def inventory(roots):
    rows=[];paths={}
    for alias,root in roots.items():
        root=Path(root)
        if not root.is_dir() or root.is_symlink():raise RuntimeError('Missing/linked export root: '+str(root))
        for directory,dirs,files in os.walk(root,followlinks=False):
            for name in list(dirs):
                p=Path(directory)/name
                if name in {'runtime_cache','__pycache__'}:dirs.remove(name)
                elif p.is_symlink():raise RuntimeError('Review linked directory before export: '+str(p))
            for name in sorted(files):
                if name.endswith('.pyc'):continue
                p=Path(directory)/name
                if not stat.S_ISREG(p.lstat().st_mode):raise RuntimeError('Review nonregular export file: '+str(p))
                key=str(Path(alias)/p.relative_to(root));before=p.stat()
                digest=sha(p);after=p.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during export inventory')
                rows.append(dict(path=key,size=after.st_size,sha256=digest));paths[key]=p
    rows.sort(key=lambda r:r['path'])
    if len({r['path'] for r in rows})!=len(rows):raise RuntimeError('Export name collision')
    return rows,paths

def export(root):
    root=Path(root).resolve()
    if root.parent!=BASE.resolve() or not root.name.startswith('run_'):raise RuntimeError('Wrong result root')
    proof=read(root/'SUCCESS.json')
    if proof['status']!='EXP6E3_MATCHED_GAMMA_SEED42_COMPLETE' or proof['optimizer_steps']!=7648 or proof['new_test_inference_calls']!=0:
        raise RuntimeError('Only completed verified scientific run can be exported')
    for f in ['run.log','exit_code.txt','best_checkpoint/model.pt','best_checkpoint/training_state.pt','validation_predictions.npz',
              'EXP6E3_TRAINING_SUMMARY_CN.md','EXP6E3_CHECKPOINT_RELOAD_VERIFICATION.json','FROZEN_POSTCHECK.json']:
        if not (root/f).is_file():raise RuntimeError('Missing required new artifact: '+f)
    if (root/'exit_code.txt').read_text().strip()!='0':raise RuntimeError('Science exit code is not0')
    for f in ['best_checkpoint/model.pt','best_checkpoint/training_state.pt']:
        if (root/f).stat().st_size<=MIN_MODEL_BYTES:raise RuntimeError('Checkpoint/state unexpectedly small')
    roots={'results/'+root.name:root,'experiments/'+HERE.name:HERE}
    for name in read(HERE/'INPUT_REFERENCE.json')['frozen_source_directories']:
        roots['experiments/'+name]=HERE.parent/name
    rows,paths=inventory(roots);size=sum(r['size'] for r in rows)
    RECORDS.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(RECORDS).free<size+512*1024**2:raise RuntimeError('Insufficient archive space; completed science preserved; no retraining')
    out=Path(tempfile.mkdtemp(prefix='export_'+root.name+'_',dir=RECORDS))
    write(out/'FILES.json',rows)
    write(out/'EXPORT_STATUS.json',dict(status='EXPORT_BUILDING',source=str(root),files=len(rows),logical_bytes=size,
        external_drive='NOT_ATTEMPTED_USER_AUTHORIZED_DEFER',mac_local='PENDING_USER_TRANSFER'))
    archive=out/'Exp6E3_results.tar.gz'
    with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
        for row in rows:
            p=paths[row['path']];info=tar.gettarinfo(str(p),arcname=row['path'])
            if info.size!=row['size'] or not info.isfile():raise RuntimeError('Source changed before tar')
            with p.open('rb') as src:tar.addfile(info,src)
    # Read the archive and hash actual archived bytes independently of source.
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if [m.name for m in members]!=[r['path'] for r in rows]:raise RuntimeError('Archive contents differ')
        for member,row in zip(members,rows):
            h=hashlib.sha256()
            with tar.extractfile(member) as src:
                for b in iter(lambda:src.read(4*1024**2),b''):h.update(b)
            if member.size!=row['size'] or h.hexdigest()!=row['sha256']:raise RuntimeError('Archived file hash differs')
    if inventory(roots)[0]!=rows:raise RuntimeError('Source changed while exporting; preserve incomplete package')
    digest=sha(archive)
    (out/'ARCHIVE.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    shutil.copyfile(HERE/'mac_unpack_verify.py',out/'mac_unpack_verify.py')
    write(out/'EXPORT_STATUS.json',dict(status='EXP6E3_RESULT_EXPORT_VERIFIED_READY_FOR_MAC',source=str(root),
        archive=str(archive),archive_sha256=digest,archive_bytes=archive.stat().st_size,files=len(rows),logical_bytes=size,
        created=datetime.now(timezone.utc).isoformat(),external_drive='NOT_ATTEMPTED_USER_AUTHORIZED_DEFER',
        mac_local='PENDING_USER_TRANSFER_AND_TARGET_SHA256_VERIFICATION',github='NOT_PUSHED',scope='Complete new run plus source dependencies; not entire historical project'))
    (BASE/'LATEST_EXP6E3_EXPORT.txt').write_text(str(out)+'\n',encoding='utf-8')
    print(json.dumps(read(out/'EXPORT_STATUS.json'),indent=2),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();export(a.root)
