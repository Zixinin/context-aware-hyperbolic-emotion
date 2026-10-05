# GoEmotions sentiment-derived hierarchy v0

**状态：用户于2026-09-23明确采用；adapter已实现，GPU运行待验收。不是GoEmotions的ground-truth hierarchy；coarse仅为派生supervision。**

[来源：Google Research sentiment_dict.json](https://github.com/google-research/google-research/blob/master/goemotions/data/sentiment_dict.json)。27类分组来自该文件；Neutral→neutral 是本实验提出的补充结构，单独标注。

ROOT 为逻辑根，不占类别ID；保留28原始ID，粗类追加为28–31。

| Fine ID | Fine label | Coarse parent | 来源/不确定性 |
|---|---|---|---|
| 0 | admiration | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 1 | amusement | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 2 | anger | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 3 | annoyance | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 4 | approval | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 5 | caring | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 6 | confusion | Ambiguous (30) | 官方sentiment分组；具体语境可跨情感倾向 |
| 7 | curiosity | Ambiguous (30) | 官方sentiment分组；具体语境可跨情感倾向 |
| 8 | desire | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 9 | disappointment | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 10 | disapproval | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 11 | disgust | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 12 | embarrassment | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 13 | excitement | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 14 | fear | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 15 | gratitude | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 16 | grief | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 17 | joy | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 18 | love | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 19 | nervousness | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 20 | optimism | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 21 | pride | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 22 | realization | Ambiguous (30) | 官方sentiment分组；具体语境可跨情感倾向 |
| 23 | relief | Positive (28) | 官方sentiment分组；具体语境可跨情感倾向 |
| 24 | remorse | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 25 | sadness | Negative (29) | 官方sentiment分组；具体语境可跨情感倾向 |
| 26 | surprise | Ambiguous (30) | 官方sentiment分组；具体语境可跨情感倾向 |
| 27 | neutral | Neutral (31) | 工程补充单子节点分支，未获独立粗类标注 |

## 候选结构

```text
ROOT
Positive: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
Negative: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
Ambiguous: confusion, curiosity, realization, surprise
Neutral: neutral
```

## 适用边界

- neutral 与 Neutral 语义/词形相近；要检查初始化共线和 residual norm。单 child 家庭不产生 sibling loss。
- desire、relief 等不依据直觉重新分类；保留官方分组来源。
- 允许同一文本同时激活多粗类。祖先标签由细类OR派生，不新造单标签任务。
- 易替换为Ekman或聚类方案，但必须更换version/hash、缓存与run目录。
- 下一阶段 loader 必须验证：28类完全覆盖、每类唯一父、无环、ID稳定、split不变。当前 JSON schema/覆盖的静态验证不等于LGSA pipeline已通过。

来源检查：本轮使用本地已归档的官方sentiment_dict.json，并核对其SHA256与数据集MANIFEST一致；远端raw下载两次超时，未宣称本轮成功重新下载。
