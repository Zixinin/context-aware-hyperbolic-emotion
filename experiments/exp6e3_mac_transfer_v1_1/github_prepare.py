"""Prepare reviewable Exp6 source/small-result export; NEVER commit or push."""
import argparse,csv,hashlib,json,re,shutil
from pathlib import Path

ALLOWED={'.py','.sh','.md','.json','.yaml','.yml','.toml','.txt','.csv'}
BINARY={'.pt','.pth','.ckpt','.bin','.safetensors','.npz','.npy','.tar','.gz','.zip','.whl'}
SECRETS=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),re.compile(rb'AKIA[0-9A-Z]{16}')]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(repo,dest):
    repo=Path(repo).resolve();dest=Path(dest).resolve()
    if dest==repo or repo in dest.parents or dest.exists():raise RuntimeError('Use a fresh staging directory OUTSIDE the source repository')
    candidates=[];excluded=[]
    for directory in sorted((repo/'experiments').glob('exp6*')):
        if not directory.is_dir():continue
        for p in sorted(directory.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(repo)
            if {'__pycache__','runtime_cache','.git','checkpoints','best_checkpoint'} & set(rel.parts) or any(v.startswith('.venv') for v in rel.parts):continue
            if p.suffix in BINARY or p.suffix not in ALLOWED or p.stat().st_size>5*1024**2:
                excluded.append(dict(path=str(rel),size=p.stat().st_size,sha256=sha(p),reason='binary/large/outside reviewed text types; preserved in full migration backup'));continue
            candidates.append(p)
    # Only published-style summaries and aggregate metrics, never raw rows/IDs/predictions.
    names={'validation_metrics.json','validation_per_label.csv','checkpoint_selection.json','TRAINING_STATE_SCOPE.json'}
    for directory in sorted((repo/'results').glob('exp6*')):
        for p in sorted(directory.rglob('*')):
            if not p.is_file():continue
            if ('source' in p.relative_to(directory).parts or 'runtime_cache' in p.parts or 'historical_validation' in p.parts):continue
            if (p.name in names or (p.suffix=='.md' and ('SUMMARY' in p.name or 'REVIEW_CN' in p.name))) and p.stat().st_size<2*1024**2:
                candidates.append(p)
    candidates=sorted(set(candidates));issues=[]
    for p in candidates:
        data=p.read_bytes()
        try:data.decode('utf-8')
        except UnicodeDecodeError:issues.append(str(p.relative_to(repo))+': non-UTF8');continue
        if p.name.startswith('.env') or any(pattern.search(data) for pattern in SECRETS):issues.append(str(p.relative_to(repo))+': possible credential; content not printed')
    if issues:raise RuntimeError('GitHub export stopped for review: '+json.dumps(issues,ensure_ascii=False))
    dest.mkdir(parents=True)
    rows=[]
    for p in candidates:
        rel=p.relative_to(repo);target=dest/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        if sha(target)!=sha(p):raise RuntimeError('Export copy mismatch')
        rows.append(dict(path=str(rel),size=p.stat().st_size,sha256=sha(p)))
    for name,items in [('GITHUB_UPLOAD_MANIFEST.csv',rows),('GITHUB_EXCLUDED_ARTIFACTS.csv',excluded)]:
        with (dest/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]) if items else ['path','size','sha256','reason']);writer.writeheader();writer.writerows(items)
    report=f'''# Exp6 GitHub待上传包（尚未提交或推送）\n\nRepository: https://github.com/Zixinin/context-aware-hyperbolic-emotion\n\n已导出{len(rows)}个源码/小型结果文件，{sum(r['size'] for r in rows)}bytes。公开仓库仅使用经过审阅的文本清单；checkpoint、原始预测/标签向量等完整产物由外接盘迁移备份保全。没有修改.gitignore、Git index/branch、创建commit或push。\n\n下一步须核对当前GitHub分支头、与远端已有文件逐项比较，再决定提交；本地origin/main可能过时。不要git add .，不要强推。检查GitHub排除清单并保留外接盘/本地大文件副本。凭本包不能声称全项目已备份到GitHub。\n\n本轮不主动发布原始数据、预测数组或环境凭据。依赖源码/发布标签先验的来源及SHA保留在对应实验manifest，实际二进制仍需迁移包。\n'''
    (dest/'GITHUB_PREPARE_REPORT.md').write_text(report,encoding='utf-8')
    print('GITHUB_EXPORT_PREPARED_NOT_COMMITTED_NOT_PUSHED')
    print('GITHUB_STAGE='+str(dest))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    prepare(a.repo,a.destination)
