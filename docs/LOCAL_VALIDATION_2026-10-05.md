# Local verification — 2026-10-05

These checks are preparation evidence, not completed HPC experiments.

- Thirteen CPU-only tests passed: frozen configs, original LGSA loader seed propagation, raw safetensors fingerprint acceptance/rejection, threshold ties, legacy sigmoid precision, zero-division/count metrics, sample SD, duplicate prediction rejection, Slurm missing/failed batch rejection, OOM-log rejection, missing-seed rejection, validation-only threshold wiring, and complete synthetic report generation.
- Python syntax and batch-shell syntax passed. Static checks found no replacement optimizer/backward/loss implementation in the wrappers. Only seed varies between full-run configs within a model family.
- Historical BERT seed42 archived artifact hashes:38 verified. Source hashes:26 verified. Historical raw predictions reproduce Micro/Macro/Rare F1 exactly within numeric tolerance; selected validation threshold remains0.25.
- Fixed0.30 regression: Micro-F1=0.5871752515392702, Macro-F1=0.48622095169381463, Rare F1=0.3600104865353015.
- Selected0.25 regression: Micro-F1=0.5822276411102345, Macro-F1=0.4873245063404651, Rare F1=0.3731939335625971.
- All106 files in the original-code snapshots match their preserved source bytes.
- Git candidates contain no nested gitlinks, model weights, datasets, large backup archives or generated scientific runs. Source-file credential-pattern scan passed.

Pending HPC evidence: current seed42 audits, newly configured infrastructure smoke, all missing full seeds, completed Slurm accounting and the six-run report. No new GPU job or multi-seed result is claimed by this document. Synthetic unit-test plots are temporary test fixtures and are not delivered as research results.
