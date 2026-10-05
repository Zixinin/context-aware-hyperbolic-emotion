"""CPU-only check of the pinned cased model's exact raw backbone tensors."""
import json
import sys
from mslib import *
from run_model import verify_raw


def main():
    require(sys.platform == 'linux', 'Run this preflight on HPC')
    require(Path(sys.executable).resolve() == PYTHONS['bert'].resolve(), 'Use the original BERT Python environment')
    load_audit()
    current_sources('bert')
    sys.path.insert(0, str(PROJECTS['bert']))
    from src.model import load_model
    snapshot = Path(read(PROJECTS['bert'] / 'assets/manifest.json')['model_snapshot'])
    _, model = load_model(str(snapshot))
    proof = verify_raw(model.bert.state_dict(), snapshot)
    require(proof['tensors_compared'] > 100, 'Unexpectedly small BERT backbone')
    print(json.dumps({'status': 'RAW_CASED_CPU_PREFLIGHT_PASS',
                      'tensors_compared': proof['tensors_compared'],
                      'legacy_layernorm_keys_verified': len(proof['legacy_layernorm_keys_verified']),
                      'raw_file_sha256': proof['raw_file_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
