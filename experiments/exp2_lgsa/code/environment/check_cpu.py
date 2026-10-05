import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path('/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions')
ENV = Path('/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo')
assert Path(sys.prefix).resolve() == ENV.resolve(), 'Wrong Python environment'
assert sys.version_info[:2] == (3, 10)
os.chdir(ROOT)
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)
info = ROOT / 'environment'
upstream = ROOT / 'original_lgsa'
pip = [sys.executable, '-m', 'pip']
head = run(['git', '-C', upstream, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert head == '4ea3dfb39dcee82654f443811471dd29be32e87e'
assert not run(['git', '-C', upstream, 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
fingerprint = (info / 'dependency-plan.fingerprint').read_text().strip()
run(pip + ['check'])
print('Checking installed packages in a fresh Python process...', flush=True)
import torch
import torch_geometric
import transformers
import datasets
import huggingface_hub
import numpy, scipy, sklearn, pandas, matplotlib, yaml, swanlab
from torch_geometric.nn import GATConv
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
assert torch.__version__ == '2.5.1+cu118' and torch.version.cuda == '11.8'
torch.set_num_threads(1)
x = torch.randn(5, 16, requires_grad=True)
edge = torch.tensor([[0, 0, 1, 1], [2, 3, 3, 4]])
y = GATConv(16, 16)(x, edge)
y.square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
sys.path.insert(0, str(upstream))
from models.prompt import Prompt
from models.loss import HierarchicalGofLoss
run([sys.executable, upstream / 'train.py', '--help'], cwd=upstream)
assert not run(['git', '-C', upstream, 'status', '--porcelain'], capture_output=True, text=True).stdout.strip(), 'Frozen upstream changed'
with (info / 'requirements-installed-cpu.txt').open('w') as stream:
    run(pip + ['freeze'], stdout=stream)
status = {'status': 'LGSA_ENV_CPU_READY', 'python': platform.python_version(),
          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'torch_geometric': torch_geometric.__version__, 'transformers': transformers.__version__,
          'datasets': datasets.__version__, 'huggingface_hub': huggingface_hub.__version__,
          'cuda_available_here': torch.cuda.is_available(), 'gpu_smoke_passed': False,
          'original_LGSA_commit': head, 'dependency_fingerprint': fingerprint,
          'cpu_gat_backward_passed': True}
(info / 'installation-cpu-ready.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status, indent=2), flush=True)
print('LGSA_ENV_CPU_READY', flush=True)
