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
