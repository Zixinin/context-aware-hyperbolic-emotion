#!/usr/bin/env bash
set -Eeuo pipefail
FYP_ROOT=/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
cd "$FYP_ROOT"
mkdir -p logs environment/ubuntu .runtime .cache
exec 9>environment/ubuntu/setup.lock
flock -n 9 || { echo 'SETUP_ALREADY_RUNNING'; exit 1; }
trap 'code=$?; echo "SETUP_FAILED exit=$code line=$LINENO"; exit "$code"' ERR

unset PYTHONPATH PYTHONHOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
export UV_PYTHON_INSTALL_DIR="$FYP_ROOT/.runtime/python"
export UV_PYTHON_BIN_DIR="$FYP_ROOT/.runtime/bin" UV_CACHE_DIR="$FYP_ROOT/.cache/uv"
export UV_HTTP_TIMEOUT=120
export HF_HOME="$FYP_ROOT/.cache/huggingface" HF_HUB_DOWNLOAD_TIMEOUT=120
export MPLCONFIGDIR="$FYP_ROOT/.cache/matplotlib" FYP_ROOT

test "$(uname -s)" = Linux
test -f experiments/exp1_bert_baseline/code/requirements-nlp.txt
test -f experiments/exp2_lgsa/code/environment/requirements-candidate.txt
if [ -d .git/info ]; then
  printf '\n/.venv-setup/\n/.venv-bert/\n/.venv-lgsa/\n/.runtime/\n' >> .git/info/exclude
fi

retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "DOWNLOAD_RETRY $attempt"
  done
  return 1
}

echo 'STAGE_1_BOOTSTRAP'
if [ ! -d .venv-setup ]; then
  python3 -m venv .venv-setup
  touch .venv-setup/.fyp-ubuntu-owned
fi
test -f .venv-setup/.fyp-ubuntu-owned
retry .venv-setup/bin/python -m pip install --index-url https://pypi.org/simple uv==0.8.22
retry .venv-setup/bin/uv python install 3.10

for environment in .venv-bert .venv-lgsa; do
  if [ ! -e "$environment" ]; then
    .venv-setup/bin/uv venv --python 3.10 --seed "$environment"
    touch "$environment/.fyp-ubuntu-owned"
  fi
  test -f "$environment/.fyp-ubuntu-owned"
  retry "$environment/bin/python" -m pip install --index-url https://pypi.org/simple pip==25.0.1
done

echo 'STAGE_2_BERT_PACKAGES'
retry .venv-bert/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
retry .venv-bert/bin/python -m pip install --index-url https://pypi.org/simple --prefer-binary -r experiments/exp1_bert_baseline/code/requirements-nlp.txt
.venv-bert/bin/python -m pip check
.venv-bert/bin/python -m pip freeze > environment/ubuntu/bert-requirements.lock.txt

echo 'STAGE_3_LGSA_PACKAGES'
retry .venv-lgsa/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
retry .venv-lgsa/bin/python -m pip install --index-url https://pypi.org/simple --prefer-binary -r experiments/exp2_lgsa/code/environment/requirements-candidate.txt matplotlib==3.9.2 huggingface-hub==0.30.2 tokenizers==0.19.1 safetensors==0.5.3 pyarrow==19.0.1 fsspec==2024.6.1
.venv-lgsa/bin/python -m pip check
.venv-lgsa/bin/python -m pip freeze > environment/ubuntu/lgsa-requirements.lock.txt

echo 'STAGE_4_PINNED_RAW_BERT_DOWNLOAD'
cat > environment/ubuntu/download_pretrained.py <<'PY_ASSETS'
import hashlib, json, os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

root=Path(os.environ['FYP_ROOT'])
revision='cd5ef92a9fb2f889e972770a36d4ed042daf221e'
target=root/'assets/pretrained/bert-base-cased'/revision
snapshot_download(
    repo_id='google-bert/bert-base-cased',
    revision=revision,
    local_dir=str(target),
    max_workers=2,
    allow_patterns=[
        'config.json','tokenizer.json','tokenizer_config.json',
        'vocab.txt','model.safetensors','pytorch_model.bin'
    ]
)
assert (target/'model.safetensors').is_file()
assert (target/'pytorch_model.bin').is_file()
tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
config=AutoConfig.from_pretrained(target,local_files_only=True)
assert config.model_type=='bert' and config.hidden_size==768 and config.num_hidden_layers==12
assert tokenizer.do_lower_case is False

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()

record={
    'status':'PINNED_RAW_BERT_ASSETS_READY',
    'repository':'google-bert/bert-base-cased',
    'revision':revision,
    'path':str(target),
    'files_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()}
}
(root/'environment/ubuntu/pretrained_assets.json').write_text(
    json.dumps(record,indent=2)+'\n'
)
print('PINNED_RAW_BERT_ASSETS_READY',target,flush=True)
PY_ASSETS
retry .venv-bert/bin/python environment/ubuntu/download_pretrained.py

echo 'STAGE_5_IMPORTS_AND_GPU_PROBES'
cat > environment/ubuntu/probe_environment.py <<'PY_PROBE'
import importlib.metadata as metadata
import json, os, platform, subprocess, sys, traceback
from pathlib import Path
import torch, transformers, datasets, numpy, scipy, sklearn, pandas, matplotlib
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from transformers import AutoTokenizer

root=Path(os.environ['FYP_ROOT'])
model=sys.argv[1]
expected={
    'bert':{
        'torch':'2.6.0+cu118',
        'transformers':'4.51.3',
        'datasets':'3.5.0'
    },
    'lgsa':{
        'torch':'2.5.1+cu118',
        'transformers':'4.44.2',
        'datasets':'2.21.0',
        'torch-geometric':'2.6.1'
    }
}[model]
assert sys.version_info[:2]==(3,10)
assert torch.version.cuda=='11.8'
assert numpy.__version__=='1.26.4'
for name,version in expected.items():
    assert metadata.version(name)==version,(name,metadata.version(name),version)

if model=='lgsa':
    import torch_geometric, yaml, swanlab
    from torch_geometric.nn import GATConv
    source=root/'experiments/exp2_lgsa/LGSA-GoEmotions/working_lgsa'
    assert (source/'models/prompt.py').is_file(),str(source)
    sys.path.insert(0,str(source))
    from models.prompt import Prompt
    from models.loss import HierarchicalGofLoss
    torch.set_num_threads(2)
    x=torch.randn(5,16,requires_grad=True)
    edge=torch.tensor([[0,0,1,1],[2,3,3,4]])
    y=GATConv(16,16)(x,edge)
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()

raw=json.loads((root/'environment/ubuntu/pretrained_assets.json').read_text())
AutoTokenizer.from_pretrained(raw['path'],local_files_only=True)
report={
    'model':model,
    'status':'PACKAGES_AND_ASSETS_READY',
    'python':platform.python_version(),
    'executable':sys.executable,
    'packages':{n:metadata.version(n) for n in expected},
    'cuda_runtime':torch.version.cuda,
    'gpu_probe_passed':False,
    'training_entrypoint_ready':False,
    'training_entrypoint_note':'Existing HPC paths/Slurm guards still require a separate Ubuntu launcher adaptation.'
}

try:
    result=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
    report['nvidia_smi']=result.stdout+result.stderr
    assert result.returncode==0,report['nvidia_smi']
    assert torch.cuda.is_available(),'torch.cuda.is_available() is False'
    x=torch.randn(32,32,device='cuda',requires_grad=True)
    y=x@x.T
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    if model=='lgsa':
        x=torch.randn(5,16,device='cuda',requires_grad=True)
        edge=torch.tensor([[0,0,1,1],[2,3,3,4]],device='cuda')
        y=GATConv(16,16).cuda()(x,edge)
        y.square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    torch.cuda.synchronize()
    report.update(
        status='PACKAGES_AND_GPU_PROBE_PASS',
        gpu_probe_passed=True,
        gpu=torch.cuda.get_device_name(0)
    )
except Exception:
    report['status']='PACKAGES_READY_GPU_BLOCKED'
    report['gpu_error']=traceback.format_exc()

(root/'environment/ubuntu'/f'{model}-environment-check.json').write_text(
    json.dumps(report,indent=2)+'\n'
)
print(json.dumps(report,indent=2),flush=True)
PY_PROBE

.venv-bert/bin/python environment/ubuntu/probe_environment.py bert
.venv-lgsa/bin/python environment/ubuntu/probe_environment.py lgsa

.venv-setup/bin/python - <<'PY_STATUS'
import json, os
from pathlib import Path

root=Path(os.environ['FYP_ROOT'])
directory=root/'environment/ubuntu'
runs=[
    json.loads((directory/f'{m}-environment-check.json').read_text())
    for m in ['bert','lgsa']
]
status=(
    'DEPENDENCIES_GPU_VERIFIED'
    if all(r['gpu_probe_passed'] for r in runs)
    else 'DEPENDENCIES_READY_GPU_BLOCKED'
)
record={
    'status':status,
    'models':runs,
    'formal_training_started':False,
    'ubuntu_training_launcher_ready':False
}
(directory/'SETUP_STATUS.json').write_text(json.dumps(record,indent=2)+'\n')
print(status,flush=True)
print('BERT activation: source '+str(root/'.venv-bert/bin/activate'))
print('LGSA activation: source '+str(root/'.venv-lgsa/bin/activate'))
print('Old HPC training entrypoints still need Ubuntu path/launcher adaptation.')
PY_STATUS

echo 'SETUP_FINISHED'
