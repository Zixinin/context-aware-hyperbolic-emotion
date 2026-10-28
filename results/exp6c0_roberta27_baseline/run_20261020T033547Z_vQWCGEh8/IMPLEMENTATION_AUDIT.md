# Exp6C-0 implementation and protocol reconciliation

Scope: only A0, RoBERTa-base + 27-label multi-label standard mean BCE, seed42, four epochs. No HypEmo code is copied or imported. No later experiment or additional seed is implemented or launched.

## Authority and workflow

The user asks for code plus commands they personally run on Ubuntu. This direct request takes priority over the PDF's preference for the agent to edit through Remote SSH. Code is prepared in the existing Mac checkout and packaged for transfer; the agent does not connect to Ubuntu. Formal GPU smoke/training run only on D3020-A3 / zixinfeng, inside the existing FYP repository, using tmux and the existing ntfy topic. No package installation or system/CUDA/driver change occurs in these scripts. Inspection precedes environment selection.

The provided experiment prompt is archived verbatim as USER_EXPERIMENT_PROMPT.md. Its experiment requirements are applied within the user's request, rather than treating the document as permission for unrelated actions.

## Fixed data protocol

Read the existing FYP official simplified train/dev/test TSVs; verify their published-in-project SHA-256 values and counts. Read the 28-label order from the SHA-256-verified canonical metadata/emotions.txt. Locate neutral by name. Retain every row with >=1 remaining positive, preserving ID/text/label and row order. Remove only neutral-only rows. Never infer a single label using argmax. Preserve original IDs, retained IDs and excluded IDs in the manifest.

Local audit: train=30,587 (24,820 single / 5,767 multi); validation=3,834 (3,122 / 712); test=3,821 (3,154 / 667). Mixed neutral+emotion retained: 1,396 / 174 / 181. Neutral-only removed: 12,823 / 1,592 / 1,606. These are local data audit results, not Ubuntu training evidence. Ubuntu independently rechecks all source files and derives the same statistics before GPU smoke.

Canonical IDs do not overlap across splits. Official data has exact text overlaps: original train/validation=41, train/test=32, validation/test=10; after filtering=32/24/7. These are inherited; do not alter the official splits or claim there are no duplicate texts. Filtering introduces no new ID or text leakage. Whole-dataset statistics include test label support as required; test metrics are never used for decisions.

## Existing CLS details and explicit reconciliation

Read-only protocol references are frozen in protocol_reference.json (seven source file hashes) and rechecked on Ubuntu. Differences cause a stop, never a silent fallback.

| Item | Existing Exp5A CLS / Exp6B reference | A0 implementation |
|---|---|---|
| Encoder | BERT-base-cased | RoBERTa-base, raw last hidden state, no pooler |
| Pooling | first token | last_hidden_state[:,0,:], RoBERTa <s> |
| Head | class weight matrix and bias; no extra projection | same structure with 27 rows, equivalent to Linear(768,27) in eval |
| Dropout | expand h to [B,L,768], independent mask per label, p=0.1 | preserve this exact FYP layout with L=27 |
| Head initialization | consume a discarded normal draw, then N(0,0.02) weight; zero bias | same RNG convention with 27 rows; no trainable query |
| Optimizer | torch.optim.AdamW, bias/LayerNorm.weight excluded from decay | same named-parameter grouping, lr=5e-5, decay=0.01, betas=(0.9,0.999), eps=1e-8 |
| Gradient normalization | (mean loss * batch size).backward; divide grads by batch size | same; accumulation=1; no drop_last |
| Gradient clipping | norm 1.0, error_if_nonfinite | same, every batch |
| Schedule | linear, ceil(total_steps * 0.1) warmup | same rule, filtered training size: ceil(30587/16)=1912 steps/epoch; 7648 total; 765 warmup |
| DataLoader | seeded torch.Generator(42) for shuffle, num_workers=0, dynamic padding, pin_memory | same; max_length=50; eval batch64 |
| Precision/determinism | FP32/eager; deterministic algorithms; TF32 off; cuDNN deterministic | same plus PYTHONHASHSEED=42 and CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python |
| Selection | validation Macro-F1 at0.30; strict greater-than | same; earliest exact tie |
| Historical threshold diagnostics | some old runners also search validation thresholds | deliberately omitted: A0 only fixed inclusive0.30; no search or tuning |
| Historical checkpoint reload | Exp5A runner checks one validation batch | strengthened per current prompt to entire filtered validation split, IDs/gold/logits/predictions/metrics |
| Metric direction | FYP uses metric(y_true,y_pred); original upstream had reversed calls | correct y_true,y_pred only; count-based recomputation cross-check |

The existing per-label dropout masks differ during training from applying one shared dropout mask to h before a conventional nn.Linear. This choice is explicit: the prompt also requires the same FYP CLS dropout/classifier structure, so A0 preserves that structure rather than silently switching to a shared-mask head. In evaluation it is a plain linear affine head. No extra learned module, dense/tanh/pooler or label attention is introduced. A1/A2 must reuse this exact path and RNG convention. If a future protocol changes to shared dropout, it is a different baseline and must not overwrite this one.

## Provenance and environment

inspect_ubuntu.py uses only standard library code; it lists existing environments and immutable local RoBERTa snapshots. Default GPU Python is the existing FYP .venv-bert/bin/python because it ran the matched four-epoch FYP protocol. It must pass actual RoBERTa loading and GPU smoke; its name alone is not evidence of compatibility. The original HypEmo torch1.13.1 environment is listed for inspection, never automatically selected. No new venv or dependencies are created automatically. Return the inspection output before GPU work if anything is missing or differs.

Version 1.1 repairs a pre-training inspection bug: v1 imported data27/NumPy at module import time, so a base Python without NumPy failed before environment discovery. Environment discovery now uses standard library imports only; canonical data audit is dispatched as a separate process to the first discovered existing candidate whose metadata reports NumPy. That audit does not select or certify the GPU training environment. If no such candidate exists, inspection stops with a specific reason rather than installing dependencies. A regression test executes the import and actual inspection control flow with python -S and verifies NumPy is never imported in the discovery process. Dataset, model, loss, seed, hyperparameters and training code are unchanged.

Snapshot paths must end in an immutable 40-character commit and have RoBERTa-base config plus original vocab/merges and pretrained weights. Loading is strictly local/offline; eager attention, no pooling layer, missing/mismatched encoder keys rejected. Unexpected unused upstream LM/pooler keys are archived in loading info. Actual model/tokenizer snapshot identity, all snapshot file SHA-256s, config, tokenizer class and special IDs are saved. No claim is made that this is the authors' historical training snapshot.

## Gates, protection and failure handling

1. Verify source/reference/data hashes and inspect environment/GPU on Ubuntu; test the existing notification channel before expensive work.
2. smoke phase: fresh output directory, actual canonical single/multi-label samples, CUDA RoBERTa forward/backward, finite nonzero encoder/head gradients, real optimizer/scheduler step, independent smoke checkpoint reload on validation only. No test inference or formal training.
3. Review smoke, then explicitly launch train against that exact run root. Recheck source/data/snapshot/environment/protected-file hashes; seed fresh from42. No smoke weights enter formal training.
4. Train four full epochs, evaluate only validation each epoch, save best only on strict improvement.
5. Construct a fresh model, reload the best weights and serialized tokenizer; infer the entire validation split; require close logits (atol=rtol=1e-6), exact IDs/gold, binary predictions and aggregate metrics. Save reload proof.
6. Exclusively create the formal test claim before the only final test call. Refuse any second formal launch or automatic retry in that directory. A failed test does not cause a second inference automatically.
7. CPU verify full saved predictions, counts, per-label metrics, fixed seven rare labels, chosen epoch and complete logs. Hash-check old research files before/after. Generate summaries and sealed artifact inventory before printing success.

Existing FYP experiments/results/outputs/data/pretrained assets/docs are read-only and hashed, excluding only new Exp6C-0 directories and ephemeral caches. Existing HypEmo upstream/source/outputs/reports and Exp6B stages/reports are also read-only and hashed when present. Protected inventories include untracked research files. No writes target MVFA. No git push/merge/reset/clean or third-party source copying is used. Shared upstream lock is opened read-only, and active GPU compute causes a stop. Cache writes are confined to the new run directory. Start/success/failure notifications are built in, bounded in time and cannot replace experiment exit codes.

Supplementary neutral-only diagnostic is explicitly skipped as optional; keeping the primary inference path narrow makes the once-only formal test straightforward to audit. No 27-vs-28 Macro-F1 delta or comparison with the original single-label 66.3% is reported as model improvement.

## Completion and limits

CPU synthetic/local-data tests and static checks do not establish Ubuntu environment compatibility or training success. Only actual GPU smoke plus four-epoch artifacts, full-validation reload, the one test pass and sealed verification justify EXP6C0_ROBERTA27_MULTILABEL_BASELINE_VERIFIED. Notifications alone do not prove completion. The scripts stop after this single-seed baseline.
