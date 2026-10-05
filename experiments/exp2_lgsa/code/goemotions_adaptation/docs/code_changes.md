# GoEmotions adapter changes (2026-09-23)

Model source remains the separate tested `working_lgsa` parent-map branch.
Frozen upstream remains `4ea3dfb39dcee82654f443811471dd29be32e87e`.
HPC tested patch candidate is `209f83762dae1b354ba40e68b6952482ecf96208`, Job 2985243.
Its exact one-line diff is already committed in working_lgsa/docs/code_changes.md;
that historical commit is not edited while its smoke is pending.

This repository is branch `goemotions-adaptation`. All adapter logic is separate
from both original_lgsa and working_lgsa; both original model and loss imports
come directly from working_lgsa without monkey patches.

| Files | Change |
| --- | --- |
| hierarchy.json / hierarchy_goemotions.py | Approved sentiment-derived v0, validation, derived multi-parent supervision |
| data_goemotions.py / prepare.py | Read original dataset assets; preserve text, IDs and splits; depth targets and prompt tokens |
| lgsa_goemotions_seed42.json | Fixed first-run budget and optimizer/threshold protocol |
| run.py | Bounded smoke and full training, fresh initialization proof, checkpoint lineage and resume |
| geometry.py / gates.py / upstream_evidence.py / accept_smoke.py | Nonzero-exit geometry, training, reload and integrity gates |
| evaluation.py / comparison.py | Fine-only metrics, validation threshold search, diagnostics, source-labelled comparison |
| preflight.py / cpu_integration.py / test_adapter.py | CPU checks and tiny random-model integration before GPU submission |
| lgsa_goemotions_chain.sbatch / submit.py | afterok upstream; smoke and full in one allocation, fail-fast |

See README.md for exact deviations from upstream and from the baseline. In
particular the new training loop has four epochs, 50-token text budget plus
prompts, linear warmup/decay and gradient clipping; these are documented
adaptation choices, not extra silent model patches. The official loss and
GAT/embedding behavior are retained. No ablation or hyperbolic code is added.

Local checks are not GPU proof. Both raw-cased BERT forward/backward and the
GoEmotions geometry gate must pass on HPC before the full run can begin.
