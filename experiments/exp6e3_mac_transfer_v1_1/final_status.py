"""Separate science and target-disk backup observations; writes no experiment file."""
import argparse,json,hashlib,shutil
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def show(root,record):
    success=read(root/'SUCCESS.json');m=read(root/'validation_metrics.json')
    reload=read(root/'EXP6E3_CHECKPOINT_RELOAD_VERIFICATION.json')
    protected=read(root/'FROZEN_POSTCHECK.json')
    backup=read(record/'VERIFICATION.json');info=read(record/'BACKUP_SOURCE_AND_DESTINATION.json')
    if success['status']!='EXP6E3_MATCHED_GAMMA_SEED42_COMPLETE' or success['optimizer_steps']!=7648:raise RuntimeError('Experiment not complete')
    if backup['status']!='EXTERNAL_DRIVE_BACKUP_PASS':raise RuntimeError('Backup not verified')
    relative=str(root.relative_to(Path(info['source_roots']['repository'])))+'/SUCCESS.json'
    entry=next(r for r in read(record/'FILES.json') if r['alias']=='repository' and r['relative']==relative)
    if entry['sha256']!=hashlib.sha256((root/'SUCCESS.json').read_bytes()).hexdigest():raise RuntimeError('Backup is not bound to this science result')
    values=dict(model='A1-C-Gamma',seed=42,gamma=0.02091538317504222,alignment='Coordinate Dot Product',new_Test=0,
        validation=m,checkpoint_reload=reload['logits_exact'],historical_protection=protected['unchanged'],
        external_drive='PASS',mac_local='PENDING_USER_TRANSFER',source=str(root),backup_destination=info['destination'])
    (record/'FINAL_STATUS.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import migrate
    if str(migrate.DEST)!=info['destination'] or migrate.mount()!=backup['mount']:raise RuntimeError('Backup mount changed before final receipt copy')
    target=migrate.safe_target('records/'+record.name+'/FINAL_STATUS.json')
    shutil.copyfile(record/'FINAL_STATUS.json',target)
    if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256((record/'FINAL_STATUS.json').read_bytes()).digest():raise RuntimeError('Final receipt copy mismatch')
    print('================================================\nEXP6E3 FINAL STATUS\n')
    print('Experiment:\n  Model: A1-C-Gamma\n  Seed: 42\n  Gamma: 0.02091538317504222\n  Alignment: Coordinate Dot Product\n  Test inference: NONE\n')
    print('Validation:')
    for label,key in [('Micro-F1','micro_f1'),('Macro-F1','macro_f1'),('Rare-F1','rare_f1'),('Macro AP','macro_average_precision')]:print(f'  {label}: {100*m[key]:.4f}%')
    print('\nVerification:\n  Checkpoint reload: '+('PASS' if reload['logits_exact'] else 'FAIL')+'\n  Historical protection: '+('PASS' if protected['unchanged'] else 'FAIL'))
    print('\nBackup:\n  External drive: PASS\n  Mac local: PENDING_USER_TRANSFER\n  Source path: '+str(root)+'\n  Backup destination: '+info['destination'])
    print('================================================')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args();show(a.root,a.record)
