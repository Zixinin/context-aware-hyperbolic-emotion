# Exp6C-0 verification

PASS: canonical split/mapping SHA-256; 27-label filtering; multi-label retention; actual CUDA RoBERTa forward/backward and nonzero finite encoder/head gradients; optimizer/scheduler; four epochs and dynamically derived steps; validation-only earliest-tie selection; full validation checkpoint reload; one formal test claim/call; canonical-ID/gold alignment; all aggregate/per-label/rare metrics independently recomputed from saved predictions; package/model/data identities; existing research file hashes unchanged.

No threshold search, extra seed, extra training, or supplementary neutral-only inference.
Independent verification command reuses predictions only; it never loads a model or calls test inference.
