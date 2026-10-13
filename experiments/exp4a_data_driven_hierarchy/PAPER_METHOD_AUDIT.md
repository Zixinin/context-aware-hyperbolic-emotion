# Exp4A method audit

Reference inspection date: 2026-10-12 (Asia/Shanghai).

Sources inspected read-only:

- https://aclanthology.org/2020.acl-main.372.pdf (section 4.2, Figure 2)
- https://raw.githubusercontent.com/google-research/google-research/master/goemotions/analyze_data.py
- https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

The paper averages judgments per comment, correlates emotion vectors, and uses Ward clustering. The official analysis script removes rows with no labels across all 28 categories before averaging; its clustering call uses condensed correlation distances with optimal leaf ordering enabled. Neutral is excluded from the 27-category clustering.

Exp4A follows the supplied specification: all raw rater rows for canonical TRAIN IDs enter the PRIMARY per-comment average, including all-zero rows. This differs from the official row filter and can dilute averages. An additional TRAIN-only sensitivity tree applies that filter, retaining exactly the same comments. PRIMARY remains fixed; sensitivity results cannot select a replacement. No full-corpus reference tree is constructed.

Clustering uses scipy.cluster.hierarchy.linkage(squareform(1-R), method='ward', optimal_ordering=True). The input is a 1-D condensed distance vector, not a matrix of distance profiles. Algorithmic optimal ordering is used identically throughout; leaves are never manually rearranged.

Ward's minimum-variance interpretation requires Euclidean distances. Direct 1-Pearson distances follow the official implementation but lack that general interpretation. Average linkage is a separate robustness analysis. Heights are algorithmic merge heights, not semantic distances. All analyses use TRAIN comments only; evaluation IDs serve solely for exclusion checks. This is a train-only paper-style analysis, not an exact reproduction of Figure 2.
