# Original Exp2 Ubuntu rerun

Retrain only the original sentiment-hierarchy LGSA Exp2 seeds42/123/2026.
Do not retrain BERT or Exp4B. Original train() is extracted and compiled with
ZERO AST changes. Original prediction gathering, geometry, optimizer, scheduler,
losses, dropout,32 auxiliary targets and54 tokens are retained. Config differs
from each same-seed historical Exp2 only by destination/execution evidence, not
hyperparameters. Text50,28 task outputs,4epochs10856updates,fixed0.30,best validation
Macro-F1 checkpoint selection remain. Legacy secondary validation-selected outputs
are retained only for source fidelity, never primary evaluation/tuning.

Ubuntu wrapper removes only the Slurm entrypoint requirement: it does not fake a
Slurm job or mutate old code. It supplies requested seed consistently to train and
loader, uses the same redundant set_seed calls as Exp2.1/Exp4B, checks197 no-pooler
raw backbone tensors and50 exact legacy LayerNorm aliases BEFORE original HGS.
The only suppressed callback is the obsolete comparison to BERT42, as in Exp2.1.
Each seed has independent audit/fresh4-update smoke/raw-reset full gates. Smoke
64train/32validation overrides are diagnostic only; full config is unchanged.

Preserve all old code, datasets, tree, HPC results, Exp4B results and prior reports.
New results are under results/exp2_ubuntu/run_UTC. Final paired read-only analysis
reuses the frozen diagnostic math with only baseline paths/provenance and report
name/status adapted. It also exports original Exp2 saved32-node loss components;
these are not directly equivalent to Exp4B56-node classification losses.

Before GPU work, verify all three completed Exp4B runs and their final diagnostic
hashes; verify frozen source/raw/data/HPC provenance and same-seed configs. After
training repeat audits. Final diagnostics verify paired IDs/gold/order and N×28,
same Ubuntu host/GPU/software/runtime defaults and shared raw-backbone digest.
Same hardware reduces a confound, but CUDA/GAT can remain nondeterministic;
same hardware is not proof of bitwise identity or statistical significance.

Numeric CSV/JSON intermediates retain typed numeric values and provenance;
sample std uses ddof1 and zero-denominator prediction ratios remain undefined.
No Excel workbook is created. Final report: comparison/EXP2_UBUNTU_VS_EXP4B_SUMMARY.md.
Historical HPC comparison remains separate, never overwritten or mixed.
Start/per-seed/success/failure ntfy with bounded retries; notification failures do
not change experiment exit status. Resume skips only verified completed stages;
incomplete directories are refused and preserved for manual diagnosis.
Stop after three seeds and same-host comparison; no ablation or tuning.
