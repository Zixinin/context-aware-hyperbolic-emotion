# Codex Research Task — Exp6D: Original HypEmo vs Exp6C Audit & Distance-weighted BCE Design

## 0. Research Objective

Current FYP:

**Context-aware Multi-label Emotion Recognition with Hyperbolic Emotion Label Embeddings**

We have completed:

- Exp6A: Original HypEmo reproduction, 27-label single-label classification.
- Exp6B Stage A: Corrected original HypEmo evaluation.
- Exp6B Stage B: BERT28 Multi-label HypEmo Formula A.
- Exp6C-0 (A0): RoBERTa27 Multi-label BCE baseline.
- Exp6C-1 (A1): RoBERTa27 + Hyperbolic Auxiliary Loss.
- Exp6C-1D: Loss, gradient, label hierarchy and geometry diagnostics.
- Exp6C-1S: Hyperbolic input scaling analysis.
- Exp6C-1S-Train: s=0.03 independent training pilot.

We recently identified an important research gap:

**The original HypEmo paper uses Hyperbolic Distance-weighted Classification Loss as a central mechanism, whereas Exp6C-1 and Exp6C-1S use ordinary BCE plus hyperbolic auxiliary supervision, without distance weighting.**

Therefore, our previous experiments do not constitute a faithful multi-label adaptation of the complete original HypEmo method.

This task has two goals:

**Part A — Comprehensive code and protocol audit**

Determine exactly how Original HypEmo differs from our Exp6C implementation.

**Part B — Design a minimal and scientifically controlled multi-label adaptation of HypEmo's distance-weighted classification loss.**

Do not train any new model during this task.

The final output must specify exactly what should be implemented and tested next, but execution requires separate user authorization.

---

## 1. Project Locations and Safety

### Original HypEmo reproduction

```text
/home/zixinfeng/Documents/HypEmo-Reproduction
```

Official upstream commit:

```text
97ec6af6b19b831f69cb35a82dc65ea1395fa157
```

Official repository:

https://github.com/dinobby/HypEmo

### Existing FYP

```text
/home/zixinfeng/Documents/context-aware-hyperbolic-emotion
```

Frozen baseline A0:

```text
results/exp6c0_roberta27_baseline/run_20261020T033547Z_vQWCGEh8
```

Frozen hyperbolic auxiliary A1:

```text
results/exp6c1_roberta27_hypaux/run_20261020T050326Z_bvdWaaRL
```

Frozen scaled A1-S:

```text
results/exp6c1s_scaled_train/run_20261022T035554Z_cZrAxW97
```

Exp6B and older experiments are also frozen.

### Mandatory rules

1. Do not modify existing model code, historical outputs, configs or checkpoints.
2. Do not modify the original HypEmo source.
3. Do not start training, new seeds or formal Test inference.
4. Do not change GPU drivers, CUDA, system Python or installed dependencies.
5. Do not tune gamma, distance scaling, thresholds or training hyperparameters using previous Test results.
6. Do not add HyCoEM, LGSA, new hierarchy, ASL, graph networks or hard-negative modules.
7. Do not automatically Git push, merge, reset or clean.
8. All reports, audit scripts and derived analysis files must be written into a new isolated directory.

Suggested output:

```text
results/exp6d_hypemo_code_audit/
    run_<timestamp>/
```

Preserve source commit IDs, relevant hashes, experiment configurations and full provenance.

If any original source/config is unavailable or differs from the expected archived run, report it instead of substituting a guessed implementation.

---

## 2. Part A — Original HypEmo vs Exp6C Full Audit

First inspect the actual source code and verified artifacts.

Do not rely solely on previous Markdown summaries.

### A1. Original HypEmo source audit

Inspect at minimum:

```text
config.py
hypemo.py
hypbert.py
models/base_models.py
models/encoders.py
models/decoders.py
layers/
manifolds/
optimizers/
util_functions.py
train.py
train_label_embedding.py
data/go_emotion/label_tree.tsv
label_tree/go_emotion.bin
```

Confirm exact filenames and locations from the frozen repository.

Read the corresponding original ACL 2023 HypEmo paper.

For every important conclusion, include source file, relevant function/class and line number where available.

### A2. Current FYP implementation audit

Inspect actual code/configs for:

```text
Exp6C-0
Exp6C-1
Exp6C-1S-Train
```

Also inspect Exp6B Formula A to understand why its distance-weighted BCE caused sample-scale imbalance.

Record the exact implementation, not just intended behavior.

### A3. Required comparison matrix

Produce a detailed table covering:

| Dimension | Original HypEmo | Exp6C-0 | Exp6C-1 | Exp6C-1S |
|---|---|---|---|---|
| Task | | | | |
| Labels | | | | |
| Training data | | | | |
| Validation/Test data | | | | |
| Backbone | | | | |
| Tokenizer | | | | |
| Max sequence length | | | | |
| CLS / Pooling | | | | |
| Classification head | | | | |
| Hyperbolic projection | | | | |
| Hyperbolic embedding dimension | | | | |
| Curvature | | | | |
| Label embedding source | | | | |
| Label embedding trainability | | | | |
| Distance-weighted loss | | | | |
| Auxiliary loss | | | | |
| Positive-label aggregation | | | | |
| Loss reduction | | | | |
| Loss coefficients | | | | |
| Optimizer(s) | | | | |
| Scheduler(s) | | | | |
| Batch size | | | | |
| Epochs | | | | |
| Learning rate | | | | |
| Gradient clipping | | | | |
| Dropout | | | | |
| Checkpoint selection | | | | |
| Evaluation metrics | | | | |
| Prediction threshold | | | | |

Classify each difference as one of:

- **Required by Multi-label task**
- **Introduced for FYP protocol consistency**
- **Inherited from HypEmo**
- **Implementation correction**
- **Potentially important unexplained deviation**

Do not assume that identical model names guarantee identical layers or hyperparameters.

### A4. Especially important: classification-head differences

Verify the original HypEmo RoBERTa representation path, including:

```text
First-token hidden state
→ Pooler linear layer
→ Activation
→ Dropout
→ Classification layer
```

Confirm the actual order and whether multiple dropout operations occur.

Compare against Exp6C's classification pathway.

Determine whether A0/A1 use a structurally different classifier.

Explain whether this difference may confound comparisons with original HypEmo.

Do not modify any classification head in this audit.

### A5. Original training protocol

Pay special attention to:

```text
Original batch size
Original 20-epoch schedule
Original Adam optimizer
Original RiemannianAdam branch optimizer
Original StepLR frequency
Original tokenizer maximum length
Original alpha/gamma interpretation
```

Distinguish paper statements from official repository defaults and actual Exp6A execution.

If paper and source code disagree, preserve both facts instead of choosing one silently.

### Part A outputs

```text
01_ORIGINAL_VS_EXP6C_COMPARISON.md
01_CODE_LEVEL_DIFFERENCES.csv
01_TRAINING_PROTOCOL_AUDIT.md
```

---

## 3. Part B — Identify the Exact HypEmo Core Method

This phase must precisely reconstruct the original mathematical objective.

### B1. Original classification objective

The published HypEmo formulation uses a distance-weighted Cross-Entropy term.

Conceptually:

\[
L_{\mathrm{original}}
=
\alpha\operatorname{Mean}(d_{\mathbb H}(z_i,e_{y_i})CE_i)
+
\gamma L_{\mathrm{aux}}
\]

Verify the actual implementation for:

- Whether distance is multiplied before reduction.
- Whether alpha is 0.9 and gamma is 0.1 in the original code.
- Whether the distance is computed for the true emotion label only.
- Whether the distance weight participates in autograd.
- Whether the distance function uses NumPy, Torch or both.
- Whether the CE term is per-sample or reduced first.
- How the auxiliary Loss is calculated and reduced.
- Whether auxiliary labels and embeddings are detached.

**Important:**

The original public `hypbert.py` implementation has used `.detach().cpu().numpy()` in its distance calculation.

Verify the exact frozen source behavior.

If distance weights are detached, explain clearly:

> Distance-weighted CE changes the weighting of classifier training examples, but does not directly backpropagate through the detached distance-weight path.

The auxiliary loss may still provide gradients to the hyperbolic projection.

Do not confuse these two gradient paths.

### B2. What is the original auxiliary loss?

Reconstruct the actual computation:

```text
HNN text embedding
→ Auxiliary decoder scores
→ NLL-like classification component
→ Label embedding dot-product component
→ Auxiliary loss
```

Clarify:

- Which components are ordinary Euclidean dot products.
- Which components use actual Poincaré geodesic distance.
- Whether the dot product is computed directly on ball coordinates.
- Whether auxiliary CE/NLL competes across all emotion labels.
- Whether original single-label CE/NLL can be extended to multi-positive samples without changing the target definition.

Do not incorrectly describe the entire auxiliary loss as a hyperbolic distance loss.

### B3. What information does Parrott actually contribute?

Verify:

```text
Parrott-style hierarchy
→ Poincaré label embedding pretraining
→ Fixed 27 emotion label embeddings
→ Downstream label-related supervision
```

Clarify whether the original downstream model directly uses explicit parent-child edges, or only uses their pretrained embedding representation.

Compare this with the current A1 implementation.

Do not invent an additional Parrott supervision branch.

### Part B outputs

```text
02_ORIGINAL_HYPEMO_LOSS_RECONSTRUCTION.md
02_GRADIENT_PATH_AUDIT.md
02_PARROTT_USAGE_AUDIT.md
```

---

## 4. Part C — Design Multi-label Distance-weighted BCE

This phase is for mathematical design and numerical validation only.

**No formal training.**

We want to test whether restoring HypEmo's original distance-weighted classification idea provides benefits on our existing RoBERTa27 multi-label task.

### C1. Freeze the task

Use the established Exp6C protocol:

```text
roberta-base
27 non-neutral emotions
GoEmotions simplified
Train: 30587
Validation: 3834
Test: 3821
Seed: 42
Max length: 50
Batch size: 16
Epochs: 4
Threshold: 0.30
Checkpoint rule: Validation Macro-F1
```

The 27-label dataset, label order, sample filtering and canonical IDs must remain identical.

Do not return to BERT28 or single-label HypEmo.

The new experimental difference should be the distance-weighted classification objective, not a simultaneous change to the task, backbone or classifier head.

### C2. Starting comparator

We currently have:

**A0**

\[
L_{\mathrm{A0}}=L_{\mathrm{BCE}}
\]

**A1**

\[
L_{\mathrm{A1}}=L_{\mathrm{BCE}}+0.1L_{\mathrm{aux}}
\]

**A1-S**

\[
L_{\mathrm{A1S}}=L_{\mathrm{BCE}}+0.1L_{\mathrm{aux}}
\]

with input scaling:

\[
h_{\mathrm{hyp}}=0.03h
\]

A1-S has the following Validation results:

```text
Micro-F1: 63.9839%
Macro-F1: 56.8995%
Rare-F1: 48.8913%
```

The A0 baseline:

```text
Micro-F1: 63.4011%
Macro-F1: 58.1356%
Rare-F1: 55.0248%
```

The next candidate must clearly declare which frozen model is its primary conceptual control:

- Unscaled A1, or
- Scaled A1-S.

**Do not change the input scaling and distance weighting simultaneously in the same causal comparison.**

For example, if using A1-S as the control, retain `s=0.03` and only introduce distance weighting.

If using A1, retain `s=1` and only introduce distance weighting.

Provide a reasoned recommendation based on previous geometry diagnostics and experiment goals.

### C3. Multi-positive distance definition

Each sample can contain multiple correct emotion labels.

Let:

\[
P_i=\{k:y_{ik}=1\}
\]

For each sample, define the positive-label Poincaré distances:

\[
d_{ik}=d_{\mathbb H}(z_i,e_k),\qquad k\in P_i
\]

Candidate direct extension:

\[
\bar d_i=\frac{1}{|P_i|}
\sum_{k\in P_i}d_{ik}
\]

The classification loss must use independent binary targets, not softmax single-label CE.

Define per-example BCE explicitly:

\[
\ell_i=\frac{1}{27}\sum_{k=1}^{27}
\mathrm{BCEWithLogits}(g_{ik},y_{ik})
\]

Then define the faithful-style candidate:

\[
L_{\mathrm{weighted}}
=
\frac1B\sum_i
\alpha\operatorname{stopgrad}(\bar d_i)\ell_i
\]

and:

\[
L_{\mathrm{total}}
=
L_{\mathrm{weighted}}
+
\gamma L_{\mathrm{aux,multi}}
\]

This is a **proposed minimal multi-label extension**, not a formula claimed by the original HypEmo paper.

Do not automatically adopt it without auditing magnitude, normalization and gradient behavior.

### C4. Very important: loss-scale analysis

Our earlier Exp6B applied distance-weighted BCE on BERT28 and observed large positive-emotion sample weights.

In the current 27-label dataset, every retained sample has at least one non-neutral positive label.

Therefore, do not reuse Exp6B's neutral-only-vs-emotion interpretation directly.

Instead audit:

1. Distribution of Poincaré distance weights on the current 27-label task.
2. Mean, median, standard deviation, percentiles and maximum weight.
3. Single-positive vs multi-positive sample weight distributions.
4. Frequent-label vs rare-label sample weight distributions.
5. Whether weight magnitude correlates with the number of positive labels.
6. Whether outliers could dominate the minibatch classification objective.
7. Whether the absolute BCE scale is magnified relative to the auxiliary objective.
8. Whether alpha/gamma retain a comparable interpretation to the original HypEmo implementation.
9. Whether gradient clipping is likely to change substantially.
10. How distance weighting changes gradients to the classifier and shared encoder.

Pay particular attention to reduction differences:

**Original single-label CE and current BCE averaged across 27 label columns do not necessarily have comparable magnitudes.**

Report numerical evidence before recommending any coefficients or normalization.

### C5. Faithful vs stabilized weighting

Compare two possible design philosophies on paper and through read-only numeric analysis:

**Option F — Faithful-style weighting**

Use the raw mean positive-label Poincaré distance as the detached sample weight.

Advantage: closer to the published HypEmo distance-weighted idea.

Risk: potentially large, variable sample weights and altered BCE/auxiliary balance.

**Option N — Controlled normalized weighting**

Retain positive-label Poincaré distance as the relative weighting signal, but normalize its magnitude using a predeclared, training-derived scale.

Advantage: may preserve baseline BCE magnitude more closely.

Risk: deviates from the original HypEmo weighting mechanism.

Do not silently replace F with N.

If suggesting Option N, specify:

- Exact normalization formula.
- Whether the normalization constant is computed only from training data.
- Whether statistics are frozen or updated.
- Whether the scale depends on current model parameters.
- How the normalization affects gradients.
- Whether it preserves relative sample weighting.
- How this differs from original HypEmo.

Do not introduce clipping, temperature, learned weighting, per-label thresholds or multiple new hyperparameters unless justified as a separate future ablation.

### C6. Numerical and gradient audit

Without training or modifying any existing checkpoint:

Use a frozen compatible model and a small deterministic set of **training-only samples** to evaluate candidate loss computations.

Prefer a fixed, recorded set of canonical IDs, covering:

```text
single-positive examples
multi-positive examples
common emotion labels
rare emotion labels
```

Compute:

```text
Ordinary BCE
Raw Poincaré positive-label distance
Mean positive-label distance
Weighted BCE
Hyperbolic auxiliary loss
Combined loss
```

Then check gradients separately for:

```text
Shared RoBERTa encoder
Classification head
Hyperbolic projection
Auxiliary decoder
```

Specifically verify:

- Finite values.
- Exact label mappings.
- Correct multi-positive aggregation.
- No false-negative treatment of additional positive labels.
- No accidental backpropagation through intentionally detached distance weights.
- Correct differentiation through BCE and auxiliary branches.
- Correct Poincaré distance computation and numerical stability.
- No mutations to the frozen checkpoint or RNG state.

**Numerical analysis of a frozen checkpoint is not proof of future trained behavior.**

Do not claim the new objective improves classification unless a separately authorized training experiment establishes that.

### Part C outputs

```text
03_MULTILABEL_DISTANCE_WEIGHTING_SPEC.md
03_DISTANCE_WEIGHT_NUMERICAL_AUDIT.md
03_LOSS_GRADIENT_ANALYSIS.md
03_CANDIDATE_METHOD_COMPARISON.md
```

---

## 5. Part D — Recommend One Next Experiment

After completing the audit, propose exactly **one** primary next training experiment.

The recommendation must answer:

1. Which existing model is the matched control: A1 or A1-S?
2. Should the initial weighting use the faithful raw distance or a scale-controlled version?
3. Why is that version justified by the actual numerical audit?
4. What single factor will change?
5. What are the expected risks?
6. Which validation metrics will determine whether the modification is promising?
7. How will we verify that the hyperbolic branch remains numerically healthy?
8. How will we distinguish distance weighting from the effect of input scaling?
9. What evidence is still missing to claim independent hyperbolic benefit?

A suitable future experiment might be named:

**Exp6D-1 — Multi-label Hyperbolic Distance-weighted BCE Pilot**

But do not implement or train it yet.

### Proposed future evaluation contract

Preserve:

```text
Same RoBERTa snapshot/tokenizer
Same data IDs
Same 27-label order
Same classification head
Same hyperbolic auxiliary objective
Same input scaling as the chosen comparator
Same optimizer and scheduler
Same four-epoch budget
Same seed42
Same fixed threshold 0.30
Same Validation Macro-F1 checkpoint rule
```

Future reporting should include:

```text
Validation Micro-F1
Validation Macro-F1
Validation Rare-F1
Macro Average Precision
Per-label Precision/Recall/F1/AP
Distance-weight distribution
Gradient and clipping statistics
Poincaré clipping/radius trajectory
```

Previously observed Test scores must not be used to select the formula or hyperparameters.

An exploratory pilot should be selected and assessed primarily using Validation. Do not automatically perform a new formal Test or launch extra seeds.

If the numeric audit identifies severe instability, recommend a simpler diagnostic instead of rushing into training.

---

## 6. Final Research Report

Produce:

```text
04_EXP6D_FINAL_AUDIT_AND_RECOMMENDATION.md
04_NEXT_EXPERIMENT_PROTOCOL.md
04_EXECUTIVE_SUMMARY_CN.md
```

The Chinese executive summary must be clear enough for an undergraduate FYP supervisor meeting.

It must answer, in plain language:

**Q1. Original HypEmo and our Exp6C differ in which important ways?**

Rank the differences by their potential relevance, while distinguishing verified facts from untested explanations.

**Q2. Did our previous Exp6C experiments actually test the core HypEmo distance-weighted classification method?**

Distinguish Exp6B, Exp6C-1 and A1-S.

**Q3. What exactly is gained and lost when single-label CE becomes multi-label BCE?**

Explain the mathematical change and the effect of multiple positive labels.

**Q4. Does our current auxiliary loss still faithfully reflect the original HypEmo auxiliary objective?**

Explain the changed reduction, multi-positive softmax behavior and gradient paths.

**Q5. Why might original HypEmo improve over RoBERTa while Exp6C does not?**

Provide evidence-supported explanations and clearly mark unresolved hypotheses.

Do not claim an experimental cause has been proven merely because implementation differs.

**Q6. How can we restore distance weighting while keeping the next experiment controlled?**

Give one specific, pre-registered proposal.

**Q7. What would be required to prove Hyperbolic Geometry itself helps?**

Distinguish extra supervision, model capacity, pretrained Parrott labels and the choice of geometry.

A future capacity/loss-matched Euclidean control is required before making geometry-specific claims.

---

## 7. Mandatory Stop Conditions

Execution order:

```text
Inspect frozen original HypEmo source
                ↓
Inspect frozen Exp6C source and artifacts
                ↓
Produce complete implementation comparison
                ↓
Reconstruct original and adapted Loss functions
                ↓
Audit candidate multi-label distance weighting
                ↓
Perform read-only numerical/gradient tests
                ↓
Recommend one controlled pilot
                ↓
STOP
```

Do not automatically:

- Start Exp6D training.
- Add distance weighting to existing A1 source.
- Reproduce additional HypEmo seeds.
- Change s or gamma.
- Tune thresholds.
- Introduce Parrott revisions.
- Add HyCoEM, contrastive learning, ASL or rare-label weighting.
- Perform formal Test inference.
- Modify frozen checkpoints or past reports.

If an implementation defect is discovered, report it separately and stop before proposing unapproved changes.

### Completion marker

Print only if the complete audit has succeeded:

```text
===================================================
EXP6D_HYPEMO_VS_EXP6C_AUDIT_COMPLETE
Original HypEmo: AUDITED
Exp6C-0 / A1 / A1-S: AUDITED
Core distance-weighted loss: RECONSTRUCTED
Multi-label distance weighting: SPECIFIED
Numerical and gradient audit: REPORTED
New training: NONE
New Test inference: NONE
Next experiment: PROPOSED, NOT EXECUTED
===================================================
```

If any phase is incomplete, print a truthful partial-completion status instead.

**Scientific principle: Understand the original method first; change only one factor in the next experiment; do not sacrifice fair comparison or reproducibility to obtain a higher score.**