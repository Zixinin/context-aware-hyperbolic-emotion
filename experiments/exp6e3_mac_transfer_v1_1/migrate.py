"""Full migration backup, independent of Git ignore rules; no deletion/push.

Call sync only while science is quiescent. Records and console logs live OUTSIDE
the source repository to avoid a self-copying or continuously changing backup.
"""
import argparse,csv,hashlib,json,os,platform,shutil,stat,subprocess,sys,tempfile,uuid,fcntl
from pathlib import Path
from datetime import datetime,timezone

REPO=Path('/home/zixinfeng/Documents/context-aware-hyperbolic-emotion')
MEDIA=Path('/media/zixinfeng/外接硬盘/fzx')
DEST=MEDIA/'FYP_context-aware-hyperbolic-emotion_verified'
RECORDS=Path('/home/zixinfeng/Documents/FYP_migration_records')
UPSTREAM=Path('/home/zixinfeng/Documents/HypEmo-Reproduction/HypEmo')
HF=Path('/home/zixinfeng/Documents/HypEmo-Reproduction/environment/huggingface')
EXP6B=Path('/home/zixinfeng/Documents/HypEmo-Exp6B')
IDENTITY='fyp-full-migration-backup-v1'

def need(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def mount():
    need(platform.system()=='Linux','Backup must run on Ubuntu, not Mac')
    need(MEDIA.is_dir(),'Declared external directory missing; inspect real mounts, do not guess another destination')
    p=subprocess.run(['findmnt','--json','--target',str(MEDIA),'--output','TARGET,SOURCE,FSTYPE,OPTIONS,MAJ:MIN'],text=True,capture_output=True)
    need(p.returncode==0,'findmnt failed: '+p.stderr)
    rows=json.loads(p.stdout).get('filesystems',[]);need(len(rows)==1,'Ambiguous external mount')
    r=rows[0];target=Path(r['target']).resolve();real=MEDIA.resolve()
    need(target!=Path('/') and (target==real or target in real.parents),'Not a dedicated external mount')
    need(r['source'].startswith('/dev/') and 'rw' in r['options'].split(',') and 'ro' not in r['options'].split(','),'External block device is not mounted read/write')
    root_dev=os.stat('/').st_dev;device=os.stat(real).st_dev
    need(device!=root_dev,'Destination is on Ubuntu root filesystem, not external disk')
    need(not MEDIA.is_symlink() and not DEST.is_symlink(),'Refuse symlink destination')
    need(os.access(real,os.W_OK),'External destination not writable')
    return dict(target=str(target),source=r['source'],fstype=r['fstype'],options=r['options'],major_minor=r['maj:min'],st_dev=device,media=str(real))

def roots():
    result={'repository':REPO,'external/HypEmo-Reproduction/HypEmo':UPSTREAM,
        'external/HypEmo-Reproduction/environment/huggingface':HF,'external/HypEmo-Exp6B':EXP6B}
    # A worktree may store .git as a pointer; preserve its actual metadata too.
    gitfile=REPO/'.git'
    if gitfile.is_file():
        text=gitfile.read_text(encoding='utf-8').strip()
        need(text.startswith('gitdir: '),'Unrecognized external Git metadata pointer')
        gitdir=(REPO/text[8:]).resolve()
        result['external/git_metadata/worktree']=gitdir
        if (gitdir/'commondir').is_file():
            result['external/git_metadata/common']=(gitdir/(gitdir/'commondir').read_text().strip()).resolve()
    need(all(p.is_dir() for p in result.values()),'A required repository/external dependency directory is missing')
    return result

def excluded(name,is_dir):
    return (is_dir and (name=='__pycache__' or name=='venv' or name.startswith('.venv'))) or (not is_dir and name.endswith('.pyc'))

def inventory(source_roots):
    files=[];links=[];omitted=[]
    allowed=[p.resolve() for p in source_roots.values()]
    def visit(path,alias,relative,ancestors):
        need('\n' not in str(path) and '\r' not in str(path),'Newline in path requires explicit backup review')
        is_dir=path.is_dir()
        if excluded(path.name,is_dir):
            omitted.append(dict(path=str(path),reason='rebuildable Python environment or bytecode only'));return
        if path.is_symlink():
            real=path.resolve(strict=True)
            need(any(real==base or base in real.parents for base in allowed),'External symlink outside reviewed source roots: '+str(path)+' -> '+str(real))
            links.append(dict(source=str(path),real_target=str(real),destination='payload/'+alias+'/'+relative,materialized=True))
        st=path.stat()
        if stat.S_ISDIR(st.st_mode):
            key=(st.st_dev,st.st_ino);need(key not in ancestors,'Symlink/directory cycle: '+str(path))
            for child in sorted(path.iterdir()):visit(child,alias,(relative+'/' if relative else '')+child.name,ancestors|{key})
        elif stat.S_ISREG(st.st_mode):
            before=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            digest=sha(path);after=path.stat()
            need(before==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Source changed while hashing: '+str(path))
            files.append(dict(alias=alias,relative=relative,source=str(path),size=st.st_size,sha256=digest,
                st_dev=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
        else:raise RuntimeError('Unsupported non-regular source: '+str(path))
    for alias,path in source_roots.items():visit(path,alias,'',set())
    need(files,'No files to back up')
    return files,links,omitted

def safe_target(relative):
    p=Path(relative)
    need(not p.is_absolute() and '..' not in p.parts,'Unsafe backup relative path')
    target=DEST/p
    for parent in [target,*target.parents]:
        if parent==DEST.parent:break
        need(not parent.is_symlink(),'Backup destination contains a symlink: '+str(parent))
    return target

def verify_files(files):
    total=0
    for r in files:
        dst=safe_target('payload/'+r['alias']+'/'+r['relative'])
        need(dst.is_file() and not dst.is_symlink() and dst.stat().st_size==r['size'],'Destination file missing/wrong size: '+str(dst))
        need(sha(dst)==r['sha256'],'Destination SHA256 mismatch: '+str(dst))
        total+=r['size']
    return dict(verified_file_count=len(files),verified_logical_bytes=total,read_destination_independently=True)

def source_still_stable(files):
    for r in files:
        st=Path(r['source']).stat()
        need((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(r['st_dev'],r['inode'],r['size'],r['mtime_ns']),'Source changed during backup: '+r['source'])

def required_check(files,phase,run_root):
    keys={r['alias']+'/'+r['relative']:r for r in files}
    for prefix in ['repository/experiments/exp6e3_matched_gamma_coordinate/',
        'repository/results/exp6e2_calibrated_geodesic_train/run_20261028T083257Z_RR6oxmlJ/',
        'repository/results/exp6d2_paired_multiseed/run_20261023T061954Z_hxtvgZQ8/']:
        need(any(k.startswith(prefix) for k in keys),'Mandatory migration source missing: '+prefix)
    for name in ['repository/results/exp6e2_calibrated_geodesic_train/run_20261028T083257Z_RR6oxmlJ/best_checkpoint/model.pt',
                 'repository/results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL/best_checkpoint/model.pt']:
        need(name in keys and keys[name]['size']>100*1024**2,'Important existing checkpoint missing or implausibly small: '+name)
    if phase=='post':
        relative=run_root.relative_to(REPO);prefix='repository/'+str(relative)+'/'
        for name in ['SUCCESS.json','best_checkpoint/model.pt','best_checkpoint/training_state.pt','config.json','training_log.csv','batch_log.csv',
                     'validation_predictions.npz','validation_per_label.csv','EXP6E3_TRAINING_SUMMARY_CN.md']:
            need(prefix+name in keys,'New trained artifact missing from backup: '+name)
        need(keys[prefix+'best_checkpoint/model.pt']['size']>100*1024**2,'New checkpoint implausibly small')

def sync(record,phase,run_root):
    mount()
    # This descriptor also keeps the verified filesystem busy during copying.
    with (MEDIA/'.fyp_verified_backup.lock').open('a+b') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another backup is using this external destination')
        return sync_locked(record,phase,run_root)

def sync_locked(record,phase,run_root):
    need(record.parent.resolve()==RECORDS.resolve() and not record.exists(),'Use a fresh external record directory; do not overwrite a prior backup receipt')
    baseline=mount();source_roots=roots()
    if DEST.exists():
        need((DEST/'BACKUP_IDENTITY.json').is_file(),'Destination exists without verified ownership marker; inspect it, no overwrite')
        identity=read(DEST/'BACKUP_IDENTITY.json')
        need(identity==dict(kind=IDENTITY,source=str(REPO),host='D3020-A3'),'Backup belongs to a different project/host')
    files,links,omitted=inventory(source_roots);required_check(files,phase,run_root)
    delta=0;changed={alias:[] for alias in source_roots}
    for r in files:
        target=safe_target('payload/'+r['alias']+'/'+r['relative'])
        if not target.is_file() or target.stat().st_size!=r['size'] or sha(target)!=r['sha256']:
            delta+=r['size'];changed[r['alias']].append(r['relative'])
    reserve=8*1024**3
    need(shutil.disk_usage(MEDIA).free>=delta+reserve,'External disk free space insufficient for incremental files plus8GiB migration reserve')
    need(mount()==baseline,'Mount changed during planning')
    DEST.mkdir(exist_ok=True)
    if not (DEST/'BACKUP_IDENTITY.json').exists():write(DEST/'BACKUP_IDENTITY.json',dict(kind=IDENTITY,source=str(REPO),host='D3020-A3'))
    record.mkdir(parents=True);write(record/'FILES.json',files)
    write(record/'SYMLINK_COVERAGE.json',links);write(record/'EXCLUSIONS.json',omitted)
    write(record/'BACKUP_SOURCE_AND_DESTINATION.json',dict(created=now(),phase=phase,source_roots={k:str(v) for k,v in source_roots.items()},
        destination=str(DEST),mount=baseline,expected_files=len(files),logical_bytes=sum(r['size'] for r in files),estimated_incremental_bytes=delta,
        reserve_bytes=reserve,rsync_delete=False,preserve_changed_versions=True,git_ignore_rules_applied=False,mac_local_backup='PENDING_USER_TRANSFER'))
    with (record/'BACKUP_MANIFEST_SHA256.txt').open('x',encoding='utf-8') as f:
        for r in files:f.write(r['sha256']+'  payload/'+r['alias']+'/'+r['relative']+'\n')
    log=record/'BACKUP_RSYNC_LOG.txt'
    for alias,source in source_roots.items():
        listing=record/(alias.replace('/','__')+'_files.nul')
        listing.write_bytes(b''.join(os.fsencode(name)+b'\0' for name in changed[alias]))
        target=safe_target('payload/'+alias);target.mkdir(parents=True,exist_ok=True)
        need(mount()==baseline,'External mount changed before rsync')
        if not changed[alias]:
            with log.open('a',encoding='utf-8') as f:f.write('No SHA256 changes for '+alias+'\n')
            continue
        # SHA256 selected the changed files; force those transfers independent of
        # mtime. This avoids openrsync's --checksum/-L interoperability failure.
        args=['rsync','-rltL','--no-perms','--no-owner','--no-group','--ignore-times','--backup',
            '--backup-dir='+str(DEST/'versions'/record.name/alias),'--from0','--files-from='+str(listing),str(source)+'/',str(target)+'/']
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps(args,ensure_ascii=False)+'\n');f.flush()
            result=subprocess.run(args,cwd=DEST,stdout=f,stderr=subprocess.STDOUT)
            f.write('RSYNC_EXIT_CODE='+str(result.returncode)+'\n')
        need(result.returncode==0,'rsync failed; see '+str(log)+'\n'+log.read_text(encoding='utf-8')[-2000:])
    need(mount()==baseline,'External mount changed during copying')
    checked=verify_files(files);source_still_stable(files)
    need(mount()==baseline,'External mount changed during destination verification')
    # Confirm no included source was added/removed while the backup ran.
    current,_,_=inventory(source_roots)
    need(current==files,'Source inventory changed during backup; preserve partial copy and rerun manually when quiescent')
    write(record/'VERIFICATION.json',dict(status='EXTERNAL_DRIVE_BACKUP_PASS',**checked,mount=baseline,completed=now(),
        mac_local_backup='PENDING_USER_TRANSFER',backup_scope='full repo incl git+all checkpoint/arrays/logs, plus reviewed external dependencies'))
    text=f'''# FYP外接盘备份验收\n\nEXTERNAL_DRIVE_BACKUP: PASS\nMAC_LOCAL_BACKUP: PENDING_USER_TRANSFER\n\nSource: {REPO}\nDestination: {DEST}\nPhase: {phase}\nVerified files: {checked['verified_file_count']}\nLogical bytes: {checked['verified_logical_bytes']}\n\n全部目标文件由外接盘独立读取并逐个SHA256核验；未使用Git ignore。包含checkpoint、预测数组、CSV、配置、源码、日志、Git元数据和外部模型/先验来源。符号链接已物化为真实文件，原路径与目标在SYMLINK_COVERAGE.json。旧目标变更版本保留在versions，无--delete。元数据记录单独保存在records，不以自引用manifest证明其自身。\n'''
    (record/'BACKUP_VERIFICATION_REPORT.md').write_text(text,encoding='utf-8')
    instructions=f'''# Mac拉取迁移副本\n\nUbuntu source: {DEST}/payload/\nExpected data files: {len(files)}\nLogical bytes: {sum(r['size'] for r in files)}\n\n在Mac执行（不会让Ubuntu主动连接Mac）：\n\n```bash\nmkdir -p /Users/fengzixin/Desktop/PSP303-FYP/迁移备份_20261029\nrsync -rlt --partial --progress ubuntu4090:{DEST}/payload/ /Users/fengzixin/Desktop/PSP303-FYP/迁移备份_20261029/payload/\nscp ubuntu4090:{DEST}/records/{record.name}/FILES.json /Users/fengzixin/Desktop/PSP303-FYP/迁移备份_20261029/FILES.json\nscp ubuntu4090:{DEST}/records/{record.name}/verify_migration.py /Users/fengzixin/Desktop/PSP303-FYP/迁移备份_20261029/verify_migration.py\npython3 /Users/fengzixin/Desktop/PSP303-FYP/迁移备份_20261029/verify_migration.py --base /Users/fengzixin/Desktop/PSP303-FYP/迁移备份_20261029 --manifest /Users/fengzixin/Desktop/PSP303-FYP/迁移备份_20261029/FILES.json\n```\n\n在Mac打印MAC_LOCAL_BACKUP_VERIFIED_PASS之前保持PENDING_USER_TRANSFER。迁移到新Ubuntu时把repository和external按来源映射恢复；旧脚本包含D3020-A3/绝对路径环境保护，必须在新机器另行环境审计，不擅自改旧封存源码来绕过保护。Python环境按已保存环境记录重建；不在本次备份步骤自动安装。\n'''
    (record/'MAC_BACKUP_TRANSFER_INSTRUCTIONS.md').write_text(instructions,encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('verify_migration.py'),record/'verify_migration.py')
    meta=DEST/'records'/record.name;meta.mkdir(parents=True,exist_ok=False)
    for p in record.iterdir():
        if p.is_file():
            shutil.copyfile(p,meta/p.name)
            need(sha(meta/p.name)==sha(p),'Backup receipt copy mismatch')
    print('EXP6E3_EXTERNAL_BACKUP_VERIFIED_PASS',flush=True)
    print('EXTERNAL_DRIVE_BACKUP: PASS\nMAC_LOCAL_BACKUP: PENDING_USER_TRANSFER',flush=True)
    print('BACKUP_DESTINATION='+str(DEST));print('BACKUP_RECORD='+str(record));print('MAC_TRANSFER_INSTRUCTIONS='+str(meta/'MAC_BACKUP_TRANSFER_INSTRUCTIONS.md'))

def verify(record):
    status=read(record/'VERIFICATION.json');need(status['status']=='EXTERNAL_DRIVE_BACKUP_PASS','No successful backup receipt')
    need(mount()==status['mount'],'Verified disk is not mounted at the same location')
    result=verify_files(read(record/'FILES.json'))
    need(result['verified_file_count']==status['verified_file_count'] and result['verified_logical_bytes']==status['verified_logical_bytes'],'Backup scope changed')
    print('EXP6E3_EXTERNAL_BACKUP_VERIFIED_PASS');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sync','verify']);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--phase',choices=['pre','post'],default='pre');p.add_argument('--run-root',type=Path,default=REPO)
    a=p.parse_args()
    try:
        if a.action=='sync':sync(a.record.resolve(),a.phase,a.run_root.resolve())
        else:verify(a.record.resolve())
    except BaseException as e:
        print('EXTERNAL_DRIVE_BACKUP: FAIL\nMAC_LOCAL_BACKUP: PENDING_USER_TRANSFER',flush=True)
        print(type(e).__name__+': '+str(e),flush=True)
        raise
