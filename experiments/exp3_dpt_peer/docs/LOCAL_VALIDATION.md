# Local preparation checks, 2026-10-10

Host: macOS authoring workspace. This is not an Ubuntu execution certificate.
No real LGSA model training or GPU run was performed here.

- CPU tests:29 passed. Cover sibling extraction, self/co-GT exclusion, neutral,
  detached hard-score sampling, top-K, exact loss mathematics, finite/backward,
  empty/all-positive families, frozen hierarchy, toy zero-lambda equivalence
  (logits/loss/gradients/RNG/optimizer), active adapter/evaluation behavior,
  all12 frozen reporting pairs and sigmoid-saturation inversion handling.
- Ten additional CPU fixture tests cover canonical data signatures, original
  JSONL serialization, no-overwrite/partial preservation, cross-split leakage,
  normalized local provenance, altered artifact rejection, recorded accounting,
  pinned raw/data revisions, reload/source/selection rejection, and full matched
  dataset/prediction ID/GT checks (including rehashed wrong prediction IDs).
- Validation-curve reader accepts native JSON epoch files, normalized Exp2 CSV,
  and new Exp3 CSV; rejects missing epochs before plotting.
- The additional trainer instrumentation test removes the one inserted logging
  statement and compares the resulting AST against the original frozen train
  function. The remaining scientific loop matches structurally.
- Frozen local source audit:29 files passed (25 adapter files and4 model files).
- Copied sampling excerpt compared equal to the fetched immutable official DPT
  source, with only terminal whitespace normalization as documented.
- Python syntax parsing passed. No pre-existing tracked file was modified.

Pending on Ubuntu: actual repository/environment/assets audit, unit certificate,
official-source certificate, real LGSA lambda0 regression, peer gradient probe,
smoke checkpoint/reload/output checks, code commit, full seed42 and final comparisons.
If Ubuntu only has the raw model/dependencies, `prepare_ubuntu_assets.py` creates
new dataset assets from the frozen official revision and verifies exact Exp2.1
text/ID/GT/order signatures before publishing a manifest. This helper has only
synthetic CPU fixture validation locally; the real Ubuntu download and signature
verification are pending. Native and normalized baseline formats are supported,
without rewriting either result format. No missing evidence is fabricated.

No code was pushed to GitHub. No training was started and no score improvement is claimed.
