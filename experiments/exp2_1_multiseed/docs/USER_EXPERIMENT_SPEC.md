对，Priority 1 的意思就是：**Experiment 1（Standard BERT）和 Experiment 2（LGSA）都使用完全相同的一组随机种子重新/补跑，然后做 paired multi-seed comparison。**

最关键的不是“LGSA 多跑几个 seed”，而是：

> **BERT seed42 vs LGSA seed42**  
> **BERT seed123 vs LGSA seed123**  
> **BERT seed2026 vs LGSA seed2026**

这样最后比较的是 **mean ± std + 每个 seed 的 paired improvement**，才能判断你现在看到的 `Macro-F1 +4.66`、`Rare F1 +16.05` 到底是不是稳定现象，而不是刚好 seed42 特别幸运。

下面这份我按照“Codex 可以直接照着执行”的标准写得很细。你可以**整段原样发给 Codex**。

# FYP Experiment 2.1 — Multi-Seed Stability Comparison

## Standard BERT vs LGSA-GoEmotions

I am working on my undergraduate Final Year Project:

**Hyperbolic Representation Learning for Multi-label Emotion Recognition**

I have already completed two experiments on GoEmotions:

### Experiment 1
**Standard BERT baseline**

### Experiment 2
**LGSA-GoEmotions with sentiment-derived hierarchy v0**

The purpose of this new experiment is **NOT to modify either model or improve performance**.

The sole purpose is:

> **Evaluate whether the improvement observed for LGSA at seed 42 is stable across multiple random seeds under a controlled and reproducible protocol.**

This experiment should therefore be treated as:

# Experiment 2.1 — Multi-Seed Stability Verification

The comparison must be scientifically fair, paired by seed, and must not introduce any new modelling changes.

---

# 0. CORE RESEARCH QUESTION

The previous seed42 results were:

## Experiment 1 — Standard BERT

Fixed threshold = 0.30:

- Micro-F1 = **58.72%**
- Macro-F1 = **48.62%**
- Rare-label mean F1 = **36.00%**
- Non-rare-label mean F1 = **52.83%**

Validation-selected threshold:

- threshold = **0.25**
- Micro-F1 = **58.22%**
- Macro-F1 = **48.73%**
- Rare-label mean F1 = **37.32%**
- Non-rare-label mean F1 = **52.54%**

---

## Experiment 2 — LGSA-GoEmotions

Fixed threshold = 0.30:

- Micro-F1 = **59.46%**
- Macro-F1 = **53.29%**
- Rare-label mean F1 = **52.05%**
- Non-rare-label mean F1 = **53.70%**

Validation-selected threshold:

- threshold = **0.30**

Therefore test metrics are identical to the fixed-threshold result.

Observed LGSA improvement at fixed threshold:

- Micro-F1: **+0.74 pp**
- Macro-F1: **+4.66 pp**
- Rare-label mean F1: **+16.05 pp**
- Non-rare-label mean F1: **+0.87 pp**

The strongest label-level improvements included:

- relief: 0.00 → 47.62
- grief: 0.00 → 40.00
- pride: 38.10 → 53.85
- nervousness: 28.57 → 39.22

However:

- relief test support = 11
- grief test support = 6

Therefore the current improvement may be sensitive to random seed.

---

# 1. PRIMARY OBJECTIVE

Run both models using the **same three random seeds**:

```text
42
123
2026
```

The comparison must be paired:

```text
BERT seed42   vs LGSA seed42
BERT seed123  vs LGSA seed123
BERT seed2026 vs LGSA seed2026
```

Do NOT compare unmatched seeds.

The final report must answer:

1. Is LGSA's Macro-F1 improvement stable across seeds?
2. Is the rare-label mean F1 improvement stable across seeds?
3. Are grief and relief improvements reproducible?
4. Does LGSA consistently trade precision for recall?
5. Are improvements broad across labels, or dominated by a few rare labels?
6. How large is seed-to-seed variance for both methods?

---

# 2. ABSOLUTE RULE: DO NOT CHANGE THE MODELS

This experiment is a **stability experiment**, not a new modelling experiment.

Do NOT introduce:

- new hierarchy
- new hyperparameters
- new optimizer
- new learning rate
- new epoch number
- new threshold search
- new loss
- class weights
- oversampling
- data augmentation
- focal loss
- per-label threshold
- new prompt
- new GNN/GAT setting
- new HGA setting
- Hyperbolic layers
- new checkpoint rule
- longer training
- early stopping
- test-set tuning

The only intended experimental variable across runs is:

> **random seed**

Within each model family, everything else must remain identical to its original experiment.

---

# 3. IMPORTANT FAIRNESS PRINCIPLE

The purpose is NOT to make BERT and LGSA use identical optimizers or identical architectures.

They should each preserve their original validated protocol.

Therefore:

## Experiment 1 must preserve the original BERT protocol.

## Experiment 2 must preserve the original LGSA protocol.

Do NOT modify Experiment 1 to imitate Experiment 2.

Do NOT modify Experiment 2 to imitate Experiment 1.

The controlled factor here is:

> each model is evaluated repeatedly under its own previously fixed protocol, using the same random seeds.

This is a repeated-model stability comparison.

---

# 4. DATASET — MUST REMAIN IDENTICAL

Use exactly the same GoEmotions dataset as before:

```python
datasets.load_dataset(
    "google-research-datasets/go_emotions",
    "simplified"
)
```

Use the official predefined splits.

Expected:

```text
Train:      43,410
Validation:  5,426
Test:        5,427
```

Do NOT:

- shuffle and recreate dataset splits
- perform cross-validation
- merge validation and training
- modify label taxonomy
- remove examples
- balance classes
- oversample rare classes

The 28 fine labels must remain identical.

---

# 5. RARE LABEL DEFINITION — FREEZE IT

Rare labels must NOT be redefined for each seed.

Use exactly the same definition as previous experiments:

> Bottom 25% of labels according to full TRAINING-set frequency.

The frozen rare-label set is:

```text
grief
pride
relief
nervousness
embarrassment
remorse
fear
```

This list must be shared by all BERT and LGSA runs.

Do NOT redefine rare labels according to:

- validation performance
- test performance
- per-seed results
- LGSA performance

---

# 6. RANDOM SEEDS

Use exactly:

```text
42
123
2026
```

For every run, set all relevant random seeds.

At minimum:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
transformers.set_seed(seed)
```

If DataLoader workers are used, ensure worker randomness is also derived reproducibly from the experiment seed.

Record whether deterministic algorithms are enabled.

Do NOT silently enable or disable additional deterministic behaviour between models or seeds.

Any unavoidable nondeterminism must be documented.

---

# 7. EXPERIMENT 1 — STANDARD BERT

Use the original Experiment 1 implementation.

Model:

```text
google-bert/bert-base-cased
```

or exactly the same resolved model revision used in Experiment 1.

Architecture:

```text
Text
→ BERT
→ standard classification head
→ 28 logits
```

Task:

```text
multi-label classification
```

Loss:

```text
BCEWithLogitsLoss
```

Do NOT use softmax or CrossEntropyLoss.

---

## Experiment 1 training configuration

Preserve exactly:

```text
Epochs: 4

Learning rate:
5e-5

Per-device batch size:
16

Effective batch size:
16

Optimizer:
AdamW

Weight decay:
0.01

Bias and LayerNorm excluded from weight decay:
yes, as in original implementation

Warmup:
10%

Scheduler:
linear decay

Gradient clipping:
1.0

Max sequence length:
50

Precision:
FP32

TF32:
disabled

Checkpoint selection:
highest validation Macro-F1 at fixed threshold 0.30
```

Do NOT change these values.

---

# 8. EXPERIMENT 2 — LGSA-GoEmotions

Use the already validated **working_lgsa patched version**.

Do NOT modify frozen upstream code beyond the already accepted parent-map patch.

Frozen upstream reference:

```text
4ea3dfb39dcee82654f443811471dd29be32e87e
```

Accepted parent-map patch commit:

```text
209f83762dae1b354ba40e68b6952482ecf96208
```

The accepted fix is:

```python
- value2slot = {child: parent for child, parent in self.path_list
+ value2slot = {child: parent for parent, child in self.path_list
```

Do NOT make any additional LGSA source-code modifications unless required for multi-seed orchestration or logging.

If orchestration changes are needed, they must not alter model mathematics.

---

# 9. LGSA HIERARCHY — FREEZE v0

Use exactly the same:

# sentiment-derived hierarchy v0

```text
ROOT
├── Positive
├── Negative
├── Ambiguous
└── Neutral
    └── neutral
```

Use the same fine-label-to-parent mapping as Experiment 2 seed42.

Important:

- Positive / Negative / Ambiguous come from the previously used sentiment grouping.
- Neutral is the existing engineering extension.
- Do NOT redesign the hierarchy.
- Do NOT introduce deeper emotion hierarchy.
- Do NOT add semantic similarity edges.
- Do NOT add Hyperbolic hierarchy.

The current experiment is only about seed stability.

---

# 10. LGSA TRAINING CONFIGURATION

Preserve exactly the Experiment 2 configuration:

```text
Backbone:
raw pretrained google-bert/bert-base-cased

Model revision:
cd5ef92a9fb2f889e972770a36d4ed042daf221e

Epochs:
4

Batch size:
16

Text token budget:
50 tokens including CLS/SEP

Maximum total length after LGSA prompts:
54 tokens

Precision:
FP32

Optimizer:
Adam

Learning rate:
1e-4

Weight decay:
0

Warmup:
10%

Scheduler:
linear decay

Gradient clipping:
1.0

LGSA components:
prompt
MLM
multi-label categorical CE
single-layer GAT
HGS
HGA

HGA weight:
1.0

HGA target angle:
10 degrees

HGA norm decay:
0.9

Checkpoint selection:
highest Validation Macro-F1 @ threshold 0.30
```

Do NOT change these settings.

---

# 11. RAW INITIALIZATION REQUIREMENT

Every complete training run must independently start from the corresponding raw pretrained BERT checkpoint.

For every BERT seed:

```text
raw pretrained BERT
→ train from scratch for that seed
```

For every LGSA seed:

```text
raw pretrained BERT
→ LGSA initialization
→ train from scratch for that seed
```

Absolutely DO NOT:

- initialize seed123 from seed42 checkpoint
- initialize seed2026 from seed123
- initialize LGSA from trained BERT baseline
- initialize full training from smoke-test weights
- reuse optimizer state
- resume another seed's checkpoint

Every seed is a fully independent experiment.

---

# 12. PARAMETER FINGERPRINT CHECK

Before each full training run, verify that the backbone originates from the intended raw pretrained checkpoint.

For LGSA, retain the existing:

```text
FRESH_RAW_CASED_INITIALIZATION_VERIFIED
```

or equivalent check.

For BERT, add an equivalent verification if it does not already exist.

Save the verification result in experiment metadata.

---

# 13. SMOKE TEST POLICY

Do NOT unnecessarily repeat a scientific smoke test for every seed if the exact code path has already been validated.

However:

If Codex modifies:

- seed handling
- Slurm orchestration
- output directory handling
- experiment runner
- config parsing

then first run one lightweight smoke test to ensure the new multi-seed infrastructure works.

The smoke test is engineering validation only.

Do NOT include smoke-test metrics in scientific analysis.

---

# 14. THRESHOLD PROTOCOL

Both models must use the same evaluation protocol as Experiment 1 and Experiment 2.

For every seed:

## Setting A — Fixed threshold

Use:

```text
threshold = 0.30
```

This is the primary comparison.

Report:

- Micro Precision
- Micro Recall
- Micro F1
- Macro Precision
- Macro Recall
- Macro F1
- Rare-label mean F1
- Non-rare-label mean F1

---

## Setting B — Validation-selected global threshold

Search ONLY on validation.

Candidates:

```text
0.05
0.10
0.15
...
0.90
```

Select:

> threshold with the highest validation Macro-F1.

Then apply the selected threshold to test predictions.

Do NOT use test labels to choose threshold.

Do NOT use per-label thresholds.

Save the complete threshold search for every seed.

---

# 15. CHECKPOINT SELECTION

For every BERT and LGSA run:

At the end of each epoch:

1. evaluate validation set
2. use fixed threshold = 0.30
3. calculate validation Macro-F1
4. save the checkpoint if it is the best so far

Select:

> checkpoint with highest validation Macro-F1 at fixed threshold 0.30

Do NOT select checkpoint according to:

- test score
- rare-label test score
- validation-selected threshold
- training loss

Record:

```text
best_epoch
best_validation_macro_f1_fixed_0.30
```

---

# 16. TEST SET RULE

Test set is FINAL EVALUATION ONLY.

For each trained seed:

1. choose checkpoint using validation
2. choose optional global threshold using validation
3. load selected checkpoint
4. run test inference
5. save raw logits/probabilities if practical
6. calculate test metrics

Do NOT repeatedly inspect test results while changing model configuration.

This multi-seed experiment uses a frozen protocol.

---

# 17. OUTPUT DIRECTORY STRUCTURE

Create a clean, explicit multi-seed structure.

Suggested:

```text
outputs/
├── bert_baseline/
│   ├── seed42/
│   ├── seed123/
│   └── seed2026/
│
├── lgsa_goemotions/
│   ├── seed42/
│   ├── seed123/
│   └── seed2026/
│
└── multiseed_comparison/
```

Do NOT overwrite the original seed42 artifacts.

If the existing seed42 artifacts are retained, keep them immutable.

---

# 18. SEED42 REUSE POLICY

There are already valid seed42 results for both experiments.

Do NOT blindly rerun seed42.

First verify whether the existing seed42 artifacts exactly satisfy the current frozen protocol.

For Experiment 1 verify at least:

```text
dataset
dataset split sizes
model name/revision
seed
epochs
learning rate
batch size
effective batch size
optimizer
weight decay
warmup
scheduler
gradient clipping
max length
precision
threshold protocol
checkpoint selection rule
rare-label definition
```

For Experiment 2 verify all corresponding LGSA settings, including:

```text
patch commit
hierarchy
HGS settings
HGA settings
prompt setup
GAT
optimizer
learning rate
model revision
```

If the existing seed42 artifacts are fully compatible:

> reuse them as the seed42 member of the multi-seed experiment.

Do NOT retrain merely for convenience.

If there is any material mismatch or missing reproducibility information:

> rerun seed42 under the frozen multi-seed protocol.

Record whether seed42 was:

```text
REUSED_VERIFIED
```

or

```text
RERUN
```

and explain why.

---

# 19. REQUIRED ARTIFACTS PER SEED

For every model × seed combination save at least:

```text
config.json
experiment_metadata.json
environment.json

training_log.csv
validation_metrics_by_epoch.csv

threshold_search.csv

test_fixed_0.30_metrics.json
test_validation_selected_metrics.json

per_label_fixed_0.30_metrics.csv
per_label_validation_selected_metrics.csv

rare_label_metrics.json

predictions or logits/probabilities if storage is reasonable

checkpoint information
SUCCESS.json
```

For LGSA also preserve:

```text
geometry_acceptance.json
```

if relevant to the current initialization.

---

# 20. PER-LABEL METRICS

For every one of the 28 labels save:

```text
label_id
label
support

precision
recall
f1

tp
fp
fn

number_predicted_positive
```

Do this for fixed threshold 0.30.

Preferably also save it for validation-selected threshold.

This is important for rare-label stability analysis.

---

# 21. PRIMARY MULTI-SEED SUMMARY

After all runs finish, create:

```text
outputs/multiseed_comparison/overall_summary.csv
```

The raw table should contain one row for every model × seed.

Example:

| model | seed | threshold_setting | threshold | micro_p | micro_r | micro_f1 | macro_p | macro_r | macro_f1 | rare_f1 | nonrare_f1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT | 42 | fixed | 0.30 | ... | ... | ... | ... | ... | ... | ... | ... |
| BERT | 123 | fixed | 0.30 | ... | ... | ... | ... | ... | ... | ... | ... |
| BERT | 2026 | fixed | 0.30 | ... | ... | ... | ... | ... | ... | ... | ... |
| LGSA | 42 | fixed | 0.30 | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Create the equivalent rows for validation-selected thresholds.

---

# 22. MEAN ± STANDARD DEVIATION

For each model calculate across the three seeds:

```text
mean
sample standard deviation
minimum
maximum
```

Use sample standard deviation:

```python
std(ddof=1)
```

Report at least:

- Micro Precision
- Micro Recall
- Micro F1
- Macro Precision
- Macro Recall
- Macro F1
- Rare-label mean F1
- Non-rare-label mean F1

Create:

```text
multiseed_mean_std.csv
```

Example final table:

| Model | Micro-F1 | Macro-F1 | Rare F1 | Non-rare F1 |
|---|---:|---:|---:|---:|
| BERT | mean ± std | mean ± std | mean ± std | mean ± std |
| LGSA | mean ± std | mean ± std | mean ± std | mean ± std |

Do this separately for:

```text
A. fixed threshold 0.30
B. validation-selected threshold
```

---

# 23. PAIRED SEED COMPARISON

This is extremely important.

Do NOT only compare the two model means.

For each seed calculate:

```text
Δ = LGSA - BERT
```

For:

- Micro-F1
- Macro-F1
- Rare-label mean F1
- Non-rare-label mean F1
- Micro Precision
- Micro Recall

Create:

```text
paired_seed_deltas.csv
```

Example:

| Seed | Δ Micro-F1 | Δ Macro-F1 | Δ Rare F1 | Δ Non-rare F1 |
|---:|---:|---:|---:|---:|
| 42 | ... | ... | ... | ... |
| 123 | ... | ... | ... | ... |
| 2026 | ... | ... | ... | ... |
| Mean Δ | ... | ... | ... | ... |

The report must explicitly state:

```text
LGSA outperformed BERT on Macro-F1 in X / 3 seeds.
LGSA outperformed BERT on Rare-label F1 in X / 3 seeds.
LGSA outperformed BERT on Micro-F1 in X / 3 seeds.
```

This is more informative than only reporting average difference.

---

# 24. DO NOT OVERCLAIM STATISTICAL SIGNIFICANCE

There are only three seeds.

Do NOT claim statistical significance from n = 3.

Do NOT automatically run and overinterpret:

- paired t-test
- Wilcoxon test

unless explicitly requested later.

For the present experiment report:

- mean
- standard deviation
- paired seed-wise deltas
- consistency across seeds

Use language such as:

```text
stable across the tested seeds
consistent positive trend
variable across seeds
high seed sensitivity
```

Do NOT write:

```text
statistically significant
proven
definitively better
```

based solely on three seeds.

---

# 25. RARE-LABEL STABILITY ANALYSIS

Create a separate analysis for:

```text
grief
pride
relief
nervousness
embarrassment
remorse
fear
```

For each label and model calculate:

```text
F1 seed42
F1 seed123
F1 seed2026

mean F1
std F1
min
max
```

Create:

```text
rare_label_seed_stability.csv
```

Example:

| Label | BERT mean±std | LGSA mean±std | Mean Δ | LGSA wins |
|---|---:|---:|---:|---:|
| grief | ... | ... | ... | X/3 |
| relief | ... | ... | ... | X/3 |
| pride | ... | ... | ... | X/3 |
| nervousness | ... | ... | ... | X/3 |
| ... | ... | ... | ... | ... |

---

# 26. GRIEF AND RELIEF — SPECIAL ANALYSIS

Because these two labels drove a large fraction of the seed42 Macro-F1 improvement, report them separately.

For every seed and model include:

```text
support
TP
FP
FN
precision
recall
F1
predicted_positive_count
```

Create:

```text
grief_relief_seed_analysis.csv
```

The report should answer:

### Grief

- Does LGSA obtain non-zero F1 in all three seeds?
- How many true positives are recovered?
- How many false positives are introduced?
- Is improvement stable or highly variable?

### Relief

Same analysis.

Do NOT judge stability using F1 alone.

Because supports are extremely small, TP/FP/FN must be shown.

---

# 27. ALL-LABEL STABILITY ANALYSIS

For all 28 labels calculate:

```text
BERT F1 mean
BERT F1 std

LGSA F1 mean
LGSA F1 std

mean delta
number of seeds where LGSA > BERT
number of seeds where LGSA < BERT
number of ties
```

Create:

```text
per_label_multiseed_comparison.csv
```

Sort primarily by:

```text
mean delta descending
```

Also generate another copy sorted by:

```text
LGSA win count
```

---

# 28. NUMBER OF IMPROVED LABELS

For each seed report:

```text
number of labels where LGSA > BERT
number where LGSA < BERT
number tied
```

Example:

```text
seed42:
18 improved
10 decreased
0 tied
```

Then report cross-seed consistency.

For example:

```text
labels improved in all 3 seeds
labels improved in 2/3 seeds
labels improved in only 1/3 seeds
labels decreased in all 3 seeds
```

This will help determine whether LGSA improvement is broad or concentrated.

---

# 29. PRECISION–RECALL ANALYSIS

Seed42 showed:

```text
BERT:
Micro Precision = 55.95
Micro Recall    = 61.78

LGSA:
Micro Precision = 53.79
Micro Recall    = 66.46
```

Therefore test whether this pattern reproduces across seeds.

For each seed calculate:

```text
Δ Micro Precision
Δ Micro Recall
```

Then answer:

> Does LGSA consistently increase recall while decreasing precision?

If this occurs in all or most seeds, report it as a stable behavioural pattern.

If not, state that seed42 was not representative.

---

# 30. VALIDATION TRAINING CURVES

For every seed save:

```text
epoch
validation micro-f1
validation macro-f1
```

Create a combined table.

Check whether:

- epoch 4 remains best across most runs
- best epoch varies by seed
- validation curves remain increasing
- LGSA appears more or less seed-sensitive than BERT

Do NOT extend training beyond 4 epochs in this experiment.

Even if validation Macro-F1 is still rising at epoch 4:

> document it only.

Do NOT change the training budget.

Longer training would be a separate experiment.

---

# 31. VALIDATION-SELECTED THRESHOLD STABILITY

For every model and seed report the selected threshold.

Example:

| Model | Seed | Selected threshold |
|---|---:|---:|
| BERT | 42 | 0.25 |
| BERT | 123 | ... |
| BERT | 2026 | ... |
| LGSA | 42 | 0.30 |
| LGSA | 123 | ... |
| LGSA | 2026 | ... |

Analyse:

- whether BERT thresholds vary strongly
- whether LGSA thresholds vary strongly
- whether threshold selection changes the model ranking

Do NOT average thresholds and reuse the average.

Each seed's threshold must be selected independently using its own validation predictions.

---

# 32. OPTIONAL BUT USEFUL: SAVE RAW TEST PREDICTIONS

If storage permits, save for every run:

```text
example_id
ground_truth multi-hot vector
logits
sigmoid probabilities
```

Do NOT need to duplicate raw text if privacy/storage is a concern.

This will allow later paired error analysis without rerunning inference.

---

# 33. SLURM DESIGN

Use the existing university HPC GPU configuration appropriate for the project.

Do NOT run full training on login node.

Prefer creating independent Slurm jobs or a safe sequential runner.

Possible structure:

```text
slurm/
    bert_seed123.sbatch
    bert_seed2026.sbatch

    lgsa_seed123.sbatch
    lgsa_seed2026.sbatch
```

If seed42 must be rerun:

```text
bert_seed42.sbatch
lgsa_seed42.sbatch
```

Alternatively, use a Slurm job array only if it does not alter logging or reproducibility.

Every run must have an unambiguous:

```text
model
seed
job_id
output directory
```

Never allow two jobs to write to the same directory.

---

# 34. RESOURCE FAIRNESS

Do not deliberately give one model more training epochs or a different dataset.

GPU runtime itself does not need to be identical because the architectures differ.

Record:

```text
GPU model
number of GPUs
elapsed time
peak memory if easily available
```

but runtime is secondary to predictive performance.

---

# 35. EXPERIMENT METADATA

For every run record:

```text
experiment name
model family
seed

Git commit hash
LGSA patch commit if applicable

Slurm Job ID
hostname
GPU
CUDA
PyTorch
Transformers
Datasets
Python

model name
model revision

dataset name
dataset config
dataset split sizes

rare-label definition

epochs
batch size
effective batch size
learning rate
optimizer
weight decay
warmup
scheduler
gradient clipping
max sequence length

threshold protocol
checkpoint selection rule

start time
end time

best epoch

status
```

For LGSA additionally record:

```text
hierarchy version
HGS configuration
HGA configuration
GAT configuration
prompt configuration
```

---

# 36. COMPLETION CHECK

Every run is considered valid only if:

```text
Slurm State = COMPLETED
ExitCode = 0:0
```

and:

- expected checkpoint exists
- expected metrics exist
- threshold search exists
- per-label metrics exist
- metadata exists
- no traceback
- no OOM
- no Killed error
- checkpoint reload succeeds where implemented

For LGSA retain relevant geometry acceptance checks.

---

# 37. AUTOMATED INTEGRITY CHECK

Create a script such as:

```text
scripts/verify_multiseed_results.py
```

It should verify:

### Expected runs

```text
BERT:
42
123
2026

LGSA:
42
123
2026
```

For every run check:

```text
required files exist
seed metadata matches directory
dataset sizes match
model revision matches expected
test examples = 5427
num labels = 28
fixed threshold = 0.30
rare label list identical
best epoch in [1,2,3,4]
metrics are finite
checkpoint reload status valid
```

Abort comparison generation if a required scientific result is missing.

Do NOT silently compare incomplete runs.

---

# 38. FINAL FIGURES

Generate clean research figures under:

```text
outputs/multiseed_comparison/figures/
```

At minimum:

## Figure 1
BERT vs LGSA:

```text
Micro-F1 mean ± std
Macro-F1 mean ± std
Rare-label mean F1 ± std
Non-rare-label mean F1 ± std
```

Use error bars representing standard deviation across seeds.

---

## Figure 2

Paired seed comparison for Macro-F1.

For each seed show:

```text
BERT → LGSA
```

This should make paired improvements visible.

---

## Figure 3

Paired seed comparison for Rare-label mean F1.

---

## Figure 4

Rare-label per-class:

```text
grief
pride
relief
nervousness
embarrassment
remorse
fear
```

Plot mean F1 ± std for BERT and LGSA.

---

## Figure 5

Per-label mean Δ F1:

```text
mean(LGSA F1 - BERT F1)
```

for all 28 labels.

---

# 39. FINAL RESEARCH SUMMARY

After all runs finish, automatically generate:

```text
outputs/multiseed_comparison/MULTISEED_SUMMARY.md
```

The document must contain the following sections.

---

## 1. Experimental protocol

State:

```text
Models:
Standard BERT
LGSA-GoEmotions sentiment hierarchy v0

Seeds:
42, 123, 2026

Dataset:
GoEmotions simplified official splits

Primary threshold:
0.30

Secondary:
validation-selected global threshold
```

---

## 2. Mean ± standard deviation

Report BERT and LGSA for:

```text
Micro Precision
Micro Recall
Micro F1
Macro Precision
Macro Recall
Macro F1
Rare-label mean F1
Non-rare-label mean F1
```

---

## 3. Paired seed improvements

For every seed report:

```text
LGSA - BERT
```

for:

```text
Micro-F1
Macro-F1
Rare F1
Non-rare F1
```

---

## 4. Stability conclusion

Explicitly answer:

```text
Did LGSA outperform BERT on Macro-F1 in all 3 seeds?

Did LGSA outperform BERT on Rare F1 in all 3 seeds?

Was the magnitude of improvement stable?

Was seed42 typical or unusually strong?
```

---

## 5. grief / relief analysis

Report:

```text
TP
FP
FN
Precision
Recall
F1
```

for each seed.

State whether the seed42 improvement reproduces.

---

## 6. Precision-recall behaviour

Determine whether:

```text
LGSA Recall > BERT Recall
```

and:

```text
LGSA Precision < BERT Precision
```

is consistent across seeds.

---

## 7. Label-level consistency

Report:

```text
number of labels improved in 3/3 seeds
number improved in 2/3
number improved in 1/3
number improved in 0/3
```

Identify:

```text
most consistently improved labels
most consistently degraded labels
```

---

## 8. Scientific interpretation

Use cautious language.

Possible conclusions depending on results:

### Case A — Strong stability

If LGSA improves Macro-F1 and Rare F1 in all three seeds with reasonably low variance:

Say:

> The positive effect observed at seed42 is reproducible across the tested random seeds. The improvement is particularly consistent for rare-label performance.

Do NOT say statistically significant.

---

### Case B — Direction stable but magnitude varies

If all three seeds improve but effect sizes vary substantially:

Say:

> LGSA shows a consistent positive direction across the tested seeds, although the magnitude of the improvement is seed-sensitive.

---

### Case C — Mixed results

If LGSA wins some seeds and loses others:

Say:

> The seed42 improvement is not fully stable across random initialization, and further investigation is needed before attributing the gain to the structured-label framework.

---

### Case D — grief/relief unstable but overall rare F1 stable

Say:

> Although the largest seed42 gains for grief and relief are unstable due to extremely low support, the aggregate rare-label improvement remains consistent across seeds.

This would still be an important result.

---

# 40. IMPORTANT INTERPRETATION BOUNDARY

Even if LGSA consistently beats BERT across seeds, do NOT conclude:

```text
HGS causes the improvement
HGA causes the improvement
the sentiment hierarchy causes the improvement
Euclidean geometry causes the improvement
```

Experiment 2 simultaneously contains:

- prompt
- MLM
- alternative classification formulation
- GAT
- HGS
- HGA
- structured hierarchy
- different optimisation settings

Therefore the correct conclusion is limited to:

> **The full LGSA adaptation shows a stable or unstable improvement relative to the Standard BERT baseline.**

Causal attribution to individual LGSA components requires later ablation experiments.

---

# 41. DO NOT START ABLATION YET

This task is ONLY:

> Multi-seed stability verification.

Do not automatically proceed to:

- HGA ablation
- HGS ablation
- GAT ablation
- hierarchy replacement
- Hyperbolic model
- deeper hierarchy
- alternative optimizer
- extended epochs

After generating the final multi-seed report:

STOP.

Wait for the next research decision.

---

# 42. EXECUTION ORDER

Follow this exact order.

## Step 1
Audit existing Experiment 1 seed42 artifacts.

Determine:

```text
REUSED_VERIFIED
or
RERUN_REQUIRED
```

---

## Step 2
Audit Experiment 2 seed42 artifacts.

Determine:

```text
REUSED_VERIFIED
or
RERUN_REQUIRED
```

---

## Step 3
Implement only the minimum infrastructure required for configurable random seed.

Do not alter mathematical behaviour.

---

## Step 4
Perform static checks.

---

## Step 5
If infrastructure changed materially, run one small smoke test.

Verify:

```text
seed propagation
output isolation
checkpoint saving
metrics generation
metadata
```

---

## Step 6
Run BERT:

```text
seed123
seed2026
```

and seed42 only if required.

---

## Step 7
Run LGSA:

```text
seed123
seed2026
```

and seed42 only if required.

---

## Step 8
Verify all six scientific model-seed records.

---

## Step 9
Generate fixed-threshold comparisons.

---

## Step 10
Generate validation-selected threshold comparisons.

---

## Step 11
Generate rare-label stability analysis.

---

## Step 12
Generate grief / relief TP-FP-FN analysis.

---

## Step 13
Generate all-label stability table.

---

## Step 14
Generate plots.

---

## Step 15
Generate:

```text
MULTISEED_SUMMARY.md
```

---

# 43. WHAT I WANT CODEX TO RETURN TO ME

After preparing the implementation, first tell me clearly:

### A. What files were added or changed

For every file explain its purpose.

### B. Whether the original seed42 runs can be safely reused

Give concrete evidence.

### C. Exact Slurm commands I need to execute on HPC

Do not tell me merely:

```text
run the experiment
```

Give exact commands.

### D. Expected output directories

### E. How I monitor the jobs

Include:

```bash
squeue -u "$USER"
```

and appropriate `tail`, `sacct`, and log-check commands.

### F. How I verify successful completion

### G. Exact command to generate the final comparison after all jobs finish

---

# 44. HPC SAFETY

Do NOT launch full BERT or LGSA training on my local Mac.

All real training must run through the university HPC GPU cluster using Slurm.

Local/Codex work should be limited to:

- editing code
- static inspection
- generating scripts
- checking configs
- lightweight CPU logic if necessary

Do not perform expensive training outside HPC.

---

# 45. SCIENTIFIC PRIORITY

The priority order is:

```text
1. Fair paired comparison
2. Exact reproducibility
3. Frozen experimental protocol
4. Seed-level traceability
5. Rare-label stability
6. Complete logging
```

The goal is NOT to produce the highest number.

The goal is to answer:

> **Was the strong LGSA improvement observed at seed42 a reproducible model effect, or was it substantially influenced by random seed?**

Do not change the experiment in an attempt to make LGSA win.

Report the result exactly as observed, even if LGSA performs worse under additional seeds.

有一个地方我特别建议你就按这份来：**先不要默认重新跑 seed42。** 让 Codex 先 audit 之前两个 seed42 的 config、commit、model revision、threshold protocol 和结果文件；如果完全一致，就直接把它们作为三种子中的第一个结果，只新增 `123` 和 `2026`。这样既科学，也不浪费 HPC。

你这一步最后最想得到的其实就是两张表：

| Model | Micro-F1 | Macro-F1 | Rare F1 |
|---|---:|---:|---:|
| BERT | mean ± std | mean ± std | mean ± std |
| LGSA | mean ± std | mean ± std | mean ± std |

以及：

| Seed | Δ Macro-F1 | Δ Rare F1 |
|---:|---:|---:|
| 42 | +4.66 | +16.05 |
| 123 | ? | ? |
| 2026 | ? | ? |

如果后两行也明显是正的，那你这轮 Experiment 2 的可信度就会一下子高很多；接下来再做 ablation，研究“**为什么会提升**”，逻辑就非常顺了。