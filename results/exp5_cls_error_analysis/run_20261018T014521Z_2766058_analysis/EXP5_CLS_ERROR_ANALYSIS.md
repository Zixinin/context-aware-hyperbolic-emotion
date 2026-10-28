# Exp5 CLS GoEmotions 错误分析

## 1. 范围与核验

只诊断正式 CLS seeds 42/123/2026，固定 sigmoid≥0.30。checkpoint 只核对封存哈希和原训练的数值重载证明，不反序列化权重，不新增推理、训练、阈值搜索或测试集调参。
官方 TSV SHA256、跨 split ID、28-label 顺序、三种子样本/真实标签与原指标均已核对。下面是已选 checkpoint 的描述性结果；test 不参与候选方法、标签对或诊断阈值的自动选择。

| Seed | Best epoch | Steps | Validation Macro-F1 (%) | Checkpoint SHA256 |
| --- | --- | --- | --- | --- |
| 42 | 4 | 10856 | 52.57 | d0572eb8676c1a7c6e9dec8038965e1934d6c27c45af37e35bc0c36415b15ef9 |
| 123 | 3 | 10856 | 51.68 | 7ca731874bc1b2d75719c1d9d2af4fa32723472f9f6e56ca37cc1b8c2414dfb5 |
| 2026 | 3 | 10856 | 51.81 | a88bba27edcfc71b95c7a661bfd7e9985146310c7ceba87d36c688db78ccd852 |

## 2. 28-label 主表

Train/Val/Test 为正例数。P/R/F1 为 test 三种子的逐种子指标均值，F1 ± 样本 SD。TP/FP/FN/Pred 为每个模型的计数均值，不是把同一测试集当成三份独立样本。完整整数计数与 validation 指标见 CSV。

| Label | Train/Val/Test | P % | R % | F1 % ±SD | TP/FP/FN/Pred 均值 | 诊断 |
| --- | --- | --- | --- | --- | --- | --- |
| admiration | 4130/488/504 | 62.76 | 75.07 | 68.35 ±0.48 | 378.3/224.7/125.7/603.0 | no_dominant_flag |
| amusement | 2328/303/264 | 77.17 | 90.03 | 83.10 ±0.45 | 237.7/70.3/26.3/308.0 | no_dominant_flag |
| anger | 1567/195/198 | 49.66 | 53.54 | 51.50 ±0.29 | 106.0/107.7/92.0/213.7 | no_dominant_flag |
| annoyance | 2470/303/320 | 34.85 | 41.77 | 37.93 ±1.13 | 133.7/250.3/186.3/384.0 | both_low |
| approval | 2939/397/351 | 37.20 | 43.87 | 39.81 ±0.15 | 154.0/269.0/197.0/423.0 | both_low |
| caring | 1087/153/135 | 41.36 | 43.95 | 42.60 ±1.00 | 59.3/84.3/75.7/143.7 | both_low |
| confusion | 1368/152/153 | 38.58 | 50.11 | 43.49 ±2.49 | 76.7/122.0/76.3/198.7 | precision_dominant |
| curiosity | 2191/248/284 | 47.50 | 62.91 | 54.05 ±2.20 | 178.7/197.7/105.3/376.3 | precision_dominant |
| desire | 641/77/83 | 53.75 | 44.58 | 48.65 ±1.23 | 37.0/32.0/46.0/69.0 | no_dominant_flag |
| disappointment | 1269/163/151 | 42.09 | 24.72 | 30.46 ±2.31 | 37.3/55.0/113.7/92.3 | recall_dominant |
| disapproval | 2022/292/267 | 38.39 | 44.82 | 41.31 ±0.80 | 119.7/192.7/147.3/312.3 | both_low |
| disgust | 793/97/123 | 51.22 | 48.78 | 49.73 ±1.15 | 60.0/58.0/63.0/118.0 | no_dominant_flag |
| embarrassment [rare] | 303/35/37 | 57.94 | 41.44 | 48.26 ±3.64 | 15.3/11.3/21.7/26.7 | recall_dominant |
| excitement | 853/96/103 | 44.53 | 44.34 | 44.36 ±4.33 | 45.7/57.7/57.3/103.3 | both_low |
| fear [rare] | 596/90/78 | 62.51 | 76.07 | 68.59 ±0.33 | 59.3/35.7/18.7/95.0 | no_dominant_flag |
| gratitude | 2662/358/352 | 90.93 | 91.86 | 91.39 ±1.00 | 323.3/32.3/28.7/355.7 | no_dominant_flag |
| grief [rare] | 77/13/6 | 16.67 | 11.11 | 13.33 ±23.09 | 0.7/0.7/5.3/1.3 | both_low; seed_sensitive |
| joy | 1452/172/161 | 58.09 | 65.42 | 61.53 ±0.55 | 105.3/76.0/55.7/181.3 | no_dominant_flag |
| love | 2086/252/238 | 75.81 | 87.25 | 81.13 ±0.92 | 207.7/66.3/30.3/274.0 | no_dominant_flag |
| nervousness [rare] | 164/21/23 | 39.41 | 40.58 | 39.57 ±7.13 | 9.3/15.3/13.7/24.7 | both_low; seed_sensitive |
| optimism | 1581/209/186 | 55.65 | 59.68 | 57.57 ±0.49 | 111.0/88.7/75.0/199.7 | no_dominant_flag |
| pride [rare] | 111/15/16 | 74.40 | 35.42 | 47.95 ±5.55 | 5.7/2.0/10.3/7.7 | recall_dominant; seed_sensitive |
| realization | 1110/127/145 | 31.53 | 20.92 | 24.67 ±2.26 | 30.3/72.7/114.7/103.0 | recall_dominant |
| relief [rare] | 153/18/11 | 55.56 | 21.21 | 28.69 ±13.36 | 2.3/2.0/8.7/4.3 | recall_dominant; seed_sensitive |
| remorse [rare] | 545/68/56 | 56.56 | 82.14 | 66.99 ±0.55 | 46.0/35.3/10.0/81.3 | no_dominant_flag |
| sadness | 1326/143/156 | 55.65 | 54.27 | 54.69 ±2.59 | 84.7/69.7/71.3/154.3 | no_dominant_flag |
| surprise | 1060/129/141 | 53.96 | 56.26 | 55.09 ±4.06 | 79.3/67.7/61.7/147.0 | no_dominant_flag |
| neutral | 14219/1766/1787 | 62.27 | 73.25 | 67.20 ±1.36 | 1309.0/797.3/478.0/2106.3 | no_dominant_flag |

诊断是透明的描述规则，不是统计检验：recall_dominant 为 R<50% 且 P−R≥10 pp；precision_dominant 对称；both_low 为 P、R 均<50% 且未满足前两类。seed_sensitive 为 F1 样本 SD≥5 pp 或种子极差≥10 pp。未标记不等于模型没有问题。
P 在零预测正例时按原协议记 0，CSV 的 precision_defined/zero_prediction_seeds 标明不可解释性；不能把它当有很多 FP。

## 3. Validation 主导的问题定位

后续假设优先依据 validation；test 仅描述既有错误。以下每项同时给出三种子计数，避免只凭总体均值下结论。

**Recall 主导：** grief：P/R=58.33/12.82%，FN/FP 均值=11.3/0.3，三种子 Pred=4/0/2；relief：P/R=38.22/20.37%，FN/FP 均值=14.3/6.0，三种子 Pred=7/9/13；disappointment：P/R=43.85/22.29%，FN/FP 均值=126.7/49.0，三种子 Pred=128/73/55；pride：P/R=86.90/44.44%，FN/FP 均值=8.3/1.0，三种子 Pred=8/7/8。

**Precision 主导：** curiosity：P/R=45.93/63.98%，FN/FP 均值=89.3/187.7，三种子 Pred=356/306/377。

**P/R 同时偏低：** realization：P/R=34.33/24.67%，FN/FP 均值=95.7/67.3，三种子 Pred=146/62/88；excitement：P/R=37.39/32.99%，FN/FP 均值=64.3/54.0，三种子 Pred=94/89/74；annoyance：P/R=31.67/41.14%，FN/FP 均值=178.3/269.0，三种子 Pred=373/389/419；approval：P/R=37.11/39.21%，FN/FP 均值=241.3/266.3，三种子 Pred=428/499/339；disapproval：P/R=40.42/40.53%，FN/FP 均值=173.7/173.7，三种子 Pred=285/323/268；confusion：P/R=39.53/43.20%，FN/FP 均值=86.3/101.3，三种子 Pred=180/174/147；disgust：P/R=43.62/45.36%，FN/FP 均值=53.0/57.3，三种子 Pred=106/110/88。

### 随机种子敏感性

| Label | Split | F1 seed42/123/2026 (%) | F1 SD pp | 正例中种子预测不一致 | 三种子共同 FN |
| --- | --- | --- | --- | --- | --- |
| grief | validation | 35.29/0.00/26.67 | 18.40 | 3/13 | 10 |
| relief | validation | 24.00/22.22/32.26 | 5.36 | 4/18 | 12 |
| disappointment | validation | 34.36/27.97/23.85 | 5.30 | 34/163 | 108 |
| embarrassment | validation | 56.25/56.72/49.23 | 4.19 | 3/35 | 16 |
| disapproval | validation | 38.13/44.88/38.21 | 3.87 | 74/292 | 138 |
| grief | test | 40.00/0.00/0.00 | 23.09 | 2/6 | 4 |
| relief | test | 15.38/28.57/42.11 | 13.36 | 3/11 | 7 |
| nervousness | test | 34.04/37.04/47.62 | 7.13 | 2/23 | 13 |
| pride | test | 41.67/52.17/50.00 | 5.55 | 2/16 | 9 |
| excitement | test | 40.38/43.72/48.98 | 4.33 | 15/103 | 50 |

只列各 split 波动最高的 5 个标签，完整标记见 label_summary.csv。三个种子不是三个独立测试集；极低支持数会放大 F1 波动，不能把 SD 当置信区间。

## 4. Rare-7 与排除 grief/relief 的 Rare-5

Rare-7 固定按 training 最低频的七类：grief、pride、relief、nervousness、embarrassment、remorse、fear。Rare-5 只排除 grief、relief，不重新选择标签。

| Split | Group | Macro P % ±SD | Macro R % ±SD | Macro F1 % ±SD | F1 seed42/123/2026 % |
| --- | --- | --- | --- | --- | --- |
| validation | rare7 | 61.61 ±8.18 | 44.82 ±1.06 | 49.33 ±2.84 | 51.78/46.22/49.99 |
| validation | rare5_excluding_grief_relief | 66.94 ±0.92 | 56.11 ±2.26 | 59.70 ±1.31 | 60.63/60.26/58.21 |
| validation | grief_relief_only | 48.28 ±27.85 | 16.60 ±7.21 | 23.41 ±10.65 | 29.65/11.11/29.46 |
| test | rare7 | 51.86 ±0.92 | 44.00 ±1.41 | 44.77 ±1.98 | 44.60/42.88/46.83 |
| test | rare5_excluding_grief_relief | 58.16 ±4.34 | 55.13 ±2.40 | 54.27 ±2.89 | 51.36/54.31/57.13 |
| test | grief_relief_only | 36.11 ±12.73 | 16.16 ±6.31 | 21.01 ±6.70 | 27.69/14.29/21.05 |

七个稀有类的逐类 P/R、整数 TP/FP/FN 和 Pred 均在 label_metrics_by_seed.csv。组内 macro 不按 support 加权；额外提供 group 内 micro，不能混用。grief/relief 只作为独立敏感性诊断，不据此主张普遍长尾改善。

## 5. 多标签相似情绪分析

先按真实标签区分 A-only、B-only、both、neither，再分别统计四种预测模式。both 中同时预测两个标签是合法命中，不算混淆。A-only 中漏 A 且多报 B 只记为“相对现有标注的交叉错误”，不能证明语义误解、标注错误或 B 在文本中绝不成立。

| Pair | Split | Gold A-only/B-only/both | Pred both 均值 | A-only 漏A多报B 均值/占比 | B-only 漏B多报A 均值/占比 | Gold both 全漏均值 |
| --- | --- | --- | --- | --- | --- | --- |
| anger/annoyance | validation | 146/254/49 | 102.7 | 20.3/13.93% | 13.0/5.12% | 13.0 |
| anger/annoyance | test | 168/290/30 | 100.3 | 19.0/11.31% | 11.3/3.91% | 6.3 |
| disappointment/sadness | validation | 150/130/13 | 18.3 | 9.7/6.44% | 3.7/2.82% | 4.3 |
| disappointment/sadness | test | 134/139/17 | 20.3 | 6.3/4.73% | 3.7/2.64% | 7.3 |
| fear/nervousness | validation | 84/15/6 | 4.3 | 0.0/0.00% | 1.0/6.67% | 3.3 |
| fear/nervousness | test | 75/20/3 | 3.3 | 1.3/1.78% | 2.0/10.00% | 1.7 |

占比分母为对应 gold-only 层样本数；0 分母显示 n.a.。pair_patterns.csv 给出每层全部计数，pair_overlap.csv 给出所有 378 对的描述统计，但不按 test 排名选择下一轮标签对。validation_pair_examples.csv 仅保存预定义三对的源顺序前 3 个交叉错误实例/方向/seed，无自动语义判定。

## 6. Validation 概率与固定阈值附近错误

只计算 validation。预定义 near-FN 为 0.25≤p<0.30，near-FP 为 0.30≤p<0.35；没有扫描、推荐或替换阈值。正负例、TP/FP/FN/TN 概率的均值和分位数，以及正负例直方图均保存在 CSV。

| Label | FN 均值 | 其中 near-FN 均值 | FP 均值 | 其中 near-FP 均值 | 近阈值错误占全部 FP+FN |
| --- | --- | --- | --- | --- | --- |
| admiration | 112.3 | 10.3 | 179.7 | 11.7 | 7.53% |
| amusement | 41.3 | 4.7 | 88.3 | 4.7 | 7.20% |
| anger | 90.7 | 6.0 | 116.7 | 15.3 | 10.29% |
| annoyance | 178.3 | 15.3 | 269.0 | 42.3 | 12.89% |
| approval | 241.3 | 15.3 | 266.3 | 42.7 | 11.42% |
| caring | 74.0 | 4.3 | 91.0 | 8.0 | 7.47% |
| confusion | 86.3 | 5.3 | 101.3 | 18.3 | 12.61% |
| curiosity | 89.3 | 11.3 | 187.7 | 24.7 | 13.00% |
| desire | 32.0 | 1.3 | 42.3 | 5.0 | 8.52% |
| disappointment | 126.7 | 4.3 | 49.0 | 8.3 | 7.21% |
| disapproval | 173.7 | 6.0 | 173.7 | 22.7 | 8.25% |
| disgust | 53.0 | 3.3 | 57.3 | 5.0 | 7.55% |
| embarrassment | 17.3 | 0.0 | 12.7 | 1.3 | 4.44% |
| excitement | 64.3 | 2.3 | 54.0 | 6.0 | 7.04% |
| fear | 32.0 | 2.0 | 22.7 | 4.0 | 10.98% |
| gratitude | 37.7 | 1.3 | 37.0 | 5.3 | 8.93% |
| grief | 11.3 | 0.0 | 0.3 | 0.3 | 2.86% |
| joy | 74.3 | 4.0 | 71.3 | 11.0 | 10.30% |
| love | 30.0 | 3.7 | 91.7 | 4.7 | 6.85% |
| nervousness | 12.3 | 0.7 | 8.7 | 1.7 | 11.11% |
| optimism | 78.3 | 5.0 | 111.0 | 19.0 | 12.68% |
| pride | 8.3 | 0.3 | 1.0 | 0.0 | 3.57% |
| realization | 95.7 | 2.0 | 67.3 | 13.7 | 9.61% |
| relief | 14.3 | 1.7 | 6.0 | 0.0 | 8.20% |
| remorse | 13.7 | 1.7 | 27.0 | 2.0 | 9.02% |
| sadness | 61.0 | 3.0 | 73.7 | 7.3 | 7.67% |
| surprise | 49.0 | 2.7 | 76.7 | 8.3 | 8.75% |
| neutral | 491.0 | 53.3 | 797.3 | 67.3 | 9.37% |

近阈值占比高提示阈值/概率尺度敏感，但不证明校准失败；远低于 0.25 的 FN 不能描述成“只差一点阈值”。validation_BCE_proxy.csv 只给已有 validation 概率上的 BCE 分解，不是训练梯度证据。

## 7. 下一轮 Euclidean 假设，不实施新实验

**ASL：** training 每标签负/正比例范围为 2.05–562.77；validation 有 4 个 recall_dominant 标签（disappointment, grief, pride, relief）。这些观测可支持检验“正负项不平衡是否影响召回”的假设，但不证明当前训练被 easy negatives 的梯度主导。[ASL 原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html) 通过非对称调制和负例概率平移降低易负例影响；原证据主要来自视觉多标签任务，不能直接保证 GoEmotions Macro/Rare-F1 提升。Precision 主导的错误也不会自动被 ASL 修复。

**Semantic Label Embedding：** 预定义三对在 validation 的单边标注层中，三种子累计有 143/2337 次“漏真实类且多报另一类”（6.12%，种子×标签对层重复观察，不是独立样本数）。这支持将这些层作为标签语义区分的待检验诊断目标。[词与标签联合嵌入工作](https://aclanthology.org/P18-1216/) 提供文本/标签共享表示的机制参考，但不是本项目的效果证明。语义相似也可能使相关标签更难区分，需同时监控合法共现与单边 FP/FN；不能把所有相似标签强制互斥。仅把自由分类器权重改名为 label embeddings 不构成新的语义机制。

两条路线应先分别与冻结 CLS 基线做受控比较，不能由这次描述性 test 分析挑损失超参、阈值或最好种子。当前分析不启动 ASL、语义嵌入或其他实验。

## 8. 缺失数据与科学边界

- 本诊断不读取或生成 training logits。标准产物没有保存它们，training 只报告标签频次，不推测训练集 P/R、概率、梯度或表征。
- checkpoint 本次不加载。哈希、配置、checkpoint 选择记录与原训练重载证明提供关联证据，不能替代新的独立 checkpoint→logits 前向复验。
- 现有 logits 不能单独解释截断、讽刺、语境或标注噪声的因果作用；无此证据就不写成已发现原因。
- 三个种子共享同一数据划分；均值与样本 SD 是有限复现性描述，不作显著性、普适性或因果主张。
- 验证/测试指标均对应各自种子按 validation Macro-F1 选出的 checkpoint，不强制取最后 epoch。
- 所有输出写入新的独立诊断目录。受保护源文件、三个原 CLS 运行目录及核验输入的文件清单和 SHA256 在前后逐一比较。
