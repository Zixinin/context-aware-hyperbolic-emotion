# Local validation receipt — 2026-10-11

Status: LOCAL_SYNTHETIC_TESTS_PASS (not Ubuntu scientific completion).

- 19 new CPU synthetic tests passed, including exact comparison with the frozen LGSA metric evaluator, actual DPT-derived term loss decomposition, FP64 gradient cosine accounting, missing/zero handling, score/activity/subset metrics, co-GT exclusion and safety guards.
- 29 existing Exp3 unit tests passed without source edits.
- Ten required figure variants (the nine requested plus frequency-versus-FP) and all Q1–Q10 report sections were generated using synthetic data. Representative bar/scatter/gradient layouts were visually checked; identical scatter coordinates are combined into one annotation to avoid overprinted labels.
- Frozen Exp3 package hashes exactly match `outputs/exp3_dpt_peer/local_unit_tests_20261010_assets_v3.json`.
- Existing independent diagnostic script SHA256 remains `260824a21def8602852ae517a97c4eb7d12abf0eb8c94c1fa14bcbba94df031b`.
- Existing validation replay script SHA256 remains `3830a9f4dd616814d54c2560a2b906a5ca4280ae2a5b1b2152f83d48d3150d36`.
- Source review confirms no optimizer construction, `.step()`, `.backward()`, checkpoint writes or test model forward pass in the new diagnostic implementation.

Not verified locally: Ubuntu artifact availability/hashes, fixed-checkpoint GPU forward/autograd results, real per-label/contribution/correlation/decision outputs. Those require the user-operated Ubuntu audit and execution gates. Synthetic figures/results must never be used as actual experiment results.

The Mac sandbox emitted font-cache permission warnings during rendering; rendering still succeeded and all ten PNG files were non-empty. These are not training/GPU errors.
