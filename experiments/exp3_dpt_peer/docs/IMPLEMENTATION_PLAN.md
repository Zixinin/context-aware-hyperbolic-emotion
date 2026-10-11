# Experiment 3 implementation plan (before model implementation)

This package prepares the user-requested seed42 pilot for execution on Ubuntu.
No GPU training is performed on the Mac. No Experiment 2 file or result is edited.
Ubuntu repository path and environment still need confirmation from terminal output.

## Frozen baseline and actual call chain

The versioned Experiment 2 adapter is `experiments/exp2_lgsa/code/goemotions_adaptation`.
The existing machine checkout is typically
`experiments/exp2_lgsa/LGSA-GoEmotions/{goemotions_adaptation,working_lgsa}`.
Audit the supplied checkout against the versioned adapter before execution.

1. `hierarchy_goemotions.load_hierarchy`: 28 fine IDs 0..27, coarse IDs 28..31,
   fine-to-parent map, slots, depths and graph edges. `targets` builds `[B,64]`
   depth supervision and `[B,28]` fine multi-hot targets. Neutral is ID27 alone.
2. `data_goemotions.encode`: original 50-token text budget, 54-token total length,
   coarse and fine task MASKs in that order. Both use the extended MASK ID.
3. `Prompt.init_embedding`: semantic label initialization -> corrected HGS ->
   `GraphEmbedding.new_embedding`, including label, depth and task MASK rows.
4. `GraphEmbedding.weight()` applies GAT to input label/depth embeddings.
   Its `raw_weight()` does not apply GAT. `OutputEmbedding.weight` is bound to
   `raw_weight`, and `OutputEmbedding.forward` calls `F.linear` with that table.
5. `Prompt.forward`: BERT contextual output -> MLM prediction transform ->
   output decoder -> task MASK logits. Classification uses label vocabulary rows
   plus `multiclass_bias`. The original returned loss is MLM + multi-label
   categorical cross-entropy (not BCE).
6. Adapter `run.train` separately calls post-GAT `weight()[-35:-3]` for HGA.
   Original total is `output.loss + gof_loss_weight * alpha(step) * geo`, with
   the existing 10%-step HGA ramp. Validation and best checkpoint use Macro-F1
   at 0.30; threshold search uses validation only, 0.05..0.90 step0.05.

## Exact tensor decision

Anchor: input `x` of `model.cls.predictions.decoder`, captured with a temporary
forward pre-hook during the EXISTING forward. Select the second task MASK:
`x[task_mask].reshape(B,2,H)[:,1,:]`, `[B,768]` for BERT base. This is the existing
MLM-transformed fine-prompt representation directly used for classification.
It receives gradients through the prediction transform, BERT and GAT prompts.
No extra forward, encoder, GAT call or random draw is added to obtain it.

Keys: the exact decoder table `model.get_output_embeddings().weight()` rows
`model.vocab_size:model.vocab_size+28`, `[28,768]`. These are HGS-structured,
PRE-GAT output representations used by the actual classifier, not the HGA tensor.
The frozen upstream `new_embedding` was created with `freeze=True`. Thus these
fine-label keys are fixed in the baseline; peer gradients primarily train the
anchor path. Do not unfreeze them or retie the decoder in this pilot.

Fine scores: task-MASK rows of `output.logits`, second depth, first28 label
vocabulary columns, plus original fine `multiclass_bias`, `[B,28]`.
Selection point: detached current scores, before any loss combination.

## Original DPT extraction and adaptation

Upstream: https://github.com/ccx06/Dual-Prompt-Tuning-for-HTC
Commit: `43e2daf6578ed6df885329cf229c3f5b749e19e7`.
`src/models/model_utils.py`: verb-logit sorting and hard-verb negative sampling
are extracted verbatim and are actually called by the new loss.
`src/models/dual_prompt_model.py`, `calculate_contrastive_loss`: the positive
normalization/cosine/temperature objective is extracted into a small helper;
its ratio of exponentials is evaluated using the equivalent logsumexp form.

Adaptations: one positive GT at a time; candidate set is its fine siblings;
exclude ALL co-GT labels; K=1; no negative prompt, negative-prompt contrastive
branch, rank loss, contextual label encoder or self-sensing task. Average over
eligible GT instances, not over DPT batch/depth. Neutral singleton skips.
An empty batch yields a differentiable zero.

`L_peer = mean_(i,y eligible)[logsumexp([cos(h_i,e_y)/tau,
cos(h_i,e_n1)/tau,...]) - cos(h_i,e_y)/tau]`.
`L_total = L_Exp2 + peer_lambda * L_peer`.
Defaults: enabled=true, lambda=0.5, temperature=0.1, top_k=1. No search.

## Files and gates

Only files under `experiments/exp3_dpt_peer/` are added. A runtime wrapper around
the existing Prompt adds peer loss to its returned scalar; original forward,
classification, MLM, HGA, optimizer and schedule remain in their existing code.
The new Ubuntu launcher reuses the original training function with instrumented
logging. It supplies Ubuntu paths explicitly, without pretending to be Slurm.

Stages: source/environment audit -> unit tests -> real lambda0 model regression
on the same model/batch/RNG -> smoke (1000/500, 1epoch) -> user-reviewed smoke
gate -> full seed42 -> paired reporting. Full entrypoint verifies evidence
fingerprints. Only full loads test; smoke/regression cannot access test labels.
Outputs are new directories. Full starts from raw pretrained weights, not smoke
or Experiment 2 trained weights. No automatic multi-seed execution.

No real regression/smoke/full pass is claimed until verified on Ubuntu. Local
working LGSA HEAD is `ba429c8984f22e32e9ec11641394fe5f1beebdc3`; the historical
HPC patch commit was `209f83762dae1b354ba40e68b6952482ecf96208`. Both are identified
as the one-line parent-map repair. Runtime acceptance uses the actual frozen
model file hashes, not an assertion that the commit IDs are identical.
