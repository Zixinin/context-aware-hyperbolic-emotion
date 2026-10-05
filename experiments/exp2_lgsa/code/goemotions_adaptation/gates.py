"""Explicit RuntimeError gates remain active even if assertions were disabled."""
import math
import subprocess
import sys
from common import ROOT, PACKAGE, PATCH_COMMIT, UPSTREAM_JOB, require, read, sha, source_record, config


def upstream_gate():
    require(sys.flags.optimize == 0, 'Python optimization disables upstream assertions; unset PYTHONOPTIMIZE')
    subprocess.run([sys.executable, str(PACKAGE / 'upstream_evidence.py'),
                    '--job', UPSTREAM_JOB, '--commit', PATCH_COMMIT], check=True)


def smoke_gate(out, manifest):
    result = read(out / 'SUCCESS.json')
    require(result == read(out / 'experiment_metadata.json'), 'Smoke final metadata mismatch')
    require(result['status'] == 'GOEMOTIONS_SMOKE_PASS' and result['phase'] == 'smoke', 'Smoke did not pass')
    require(result['config_sha256'] == sha(PACKAGE / 'lgsa_goemotions_seed42.json'), 'Config changed after smoke')
    require(result['dataset_manifest_sha256'] == sha(ROOT / 'assets/goemotions-v0/manifest.json'), 'Data changed after smoke')
    require(result['adapter_source'] == source_record(), 'Adapter changed after smoke')
    require(result['model'] == manifest['model'], 'Backbone changed after smoke')
    require(result['train_examples'] == 1000 and result['validation_examples'] == 500, 'Wrong smoke sample sizes')
    require(result['test_used'] is False, 'Smoke must not evaluate test data')
    require(result['optimizer_steps'] == math.ceil(1000 / config()['batch_size']), 'Incomplete smoke')
    for field in ['checkpoint_reload_logits_match', 'backbone_gradient_passed', 'gat_gradient_passed', 'finite_training_passed']:
        require(result[field] is True, 'Smoke check failed: ' + field)
    require(math.isfinite(result['backbone_parameter_max_change']) and result['backbone_parameter_max_change'] > 0,
            'No finite backbone update')
    require(result['initialization'] == 'raw_pretrained' and result['initial_checkpoint'] is None, 'Smoke inherited trained weights')
    geometry = read(out / 'geometry_acceptance.json')
    require(geometry['status'] == 'GOEMOTIONS_HGS_PASS', 'GoEmotions geometry failed')
    require(geometry['map_matches'] is True and geometry['fine_vectors_changed'] is True, 'GoEmotions HGS map failed')
    require(geometry['top_max_abs_cosine'] <= config()['geometry_cosine_tolerance'], 'Top geometry failed')
    for f in geometry['families'].values():
        require(f['rank'] == f['children'] and f['max_abs_cosine'] <= config()['geometry_cosine_tolerance'], 'Family geometry failed')
    required = {'checkpoint_best.pt', 'checkpoint_reload.json', 'geometry_acceptance.json',
                'validation_fixed_0.30_metrics.json', 'sample_ids.json', 'steps.jsonl'}
    require(required <= set(result['artifacts_sha256']), 'Smoke evidence missing')
    for name, digest in result['artifacts_sha256'].items():
        require(sha(out / name) == digest, 'Smoke artifact changed: ' + name)
    return result
