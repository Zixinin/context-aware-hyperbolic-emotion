"""Verify candidate environment on an allocated HPC GPU; not official LGSA smoke."""
import json
import os
import platform
from pathlib import Path
import torch
import torch_geometric
import transformers
import datasets
from torch_geometric.nn import GATConv
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead

assert os.environ.get("SLURM_JOB_ID"), "Need a Slurm allocation"
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.version.cuda == "11.8", torch.version.cuda
x = torch.randn(5, 16, device="cuda", requires_grad=True)
edge = torch.tensor([[0, 0, 1, 1], [2, 3, 3, 4]], device="cuda")
gat = GATConv(16, 16, heads=1).cuda()
y = gat(x, edge)
y.square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
torch.cuda.synchronize()
report = {"status": "ENVIRONMENT_GPU_PROBE_PASS", "not_official_smoke": True,
          "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": platform.node(),
          "python": platform.python_version(), "torch": torch.__version__,
          "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
          "torch_geometric": torch_geometric.__version__, "transformers": transformers.__version__,
          "datasets": datasets.__version__, "gat_backward_finite": True}
out = Path(__file__).parent / f"runtime-{os.environ['SLURM_JOB_ID']}.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
