# Original code provenance

Repository: https://github.com/ccx06/Dual-Prompt-Tuning-for-HTC

Immutable commit: `43e2daf6578ed6df885329cf229c3f5b749e19e7` (verified from GitHub API).

`dpt_sampling.py` is a verbatim excerpt from the beginning of
[src/models/model_utils.py](https://github.com/ccx06/Dual-Prompt-Tuning-for-HTC/blob/43e2daf6578ed6df885329cf229c3f5b749e19e7/src/models/model_utils.py#L1-L80),
through `negative_sampling`. The new loss EXECUTES its `hard-verb` branch;
it is not merely an unused reference. Original author: Sishi Xiong et al.

`dpt_positive.py` extracts the positive contrastive kernel from
[calculate_contrastive_loss](https://github.com/ccx06/Dual-Prompt-Tuning-for-HTC/blob/43e2daf6578ed6df885329cf229c3f5b749e19e7/src/models/dual_prompt_model.py#L277-L290).
Original operations are normalize(query), normalize(positive keys),
normalize(negative keys), cosine positive/negative similarities, then
`-log(sum(exp(pos/tau))/(sum(exp(pos/tau))+sum(exp(neg/tau))))`.

Changes to that kernel: argument names, already-stacked tensors instead of
lists, removal of the negative MASK key (LGSA has no negative prompt), and
algebraically equivalent logsumexp evaluation to avoid exponential overflow.
Return cosine values for diagnostic logging. The peer module calls this kernel
once per eligible GT and averages those terms. This differs from DPT's joint
multi-positive numerator and batch/depth reduction. All changes are intentional.

Additional source/paper discrepancy: the released `forward` call at lines507..510
passes `label_token_repr=neg_outputs_at_mask`, although the positive kernel and
paper describe a positive-query representation. This pilot uses LGSA's actual
fine-classification anchor explicitly; it does not copy that dual-prompt caller.
The released function also accumulates `all_pos_emb/all_neg_emb` across depths.
The adaptation builds keys independently for each fine GT instance, so neither
that accumulation nor the full negative-prompt branch is imported.

Caller adaptations to sampling: the supplied `depth2label` has a single entry
containing one frozen fine-label sibling family; positives contain the entire
sample GT; scores are detached; sample_num_list is [K]. The unchanged original
function then ranks, removes GT and selects the first K valid labels.
No full DPT architecture, environment, second encoder, rank loss or self-sensing
module is imported. This is a DPT-derived peer-loss adaptation, not full DPT.

Paper: Xiong et al., Findings of ACL2024, sections3.3 and3.5,
https://aclanthology.org/2024.findings-acl.723/ . The copied snippets retain
upstream spelling and whitespace. `verify_upstream.py` compares them against
the immutable upstream file; run it on Ubuntu before acceptance. A local check
does not constitute an Ubuntu/GPU execution claim.
