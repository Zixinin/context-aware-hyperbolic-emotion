# Exp6C-0 dataset audit

official simplified splits; >=1 non-neutral positive; 27 multi-hot

Canonical mapping is read and SHA-256 checked from metadata/emotions.txt; neutral is removed by name.
Original/retained/excluded canonical IDs and row/target hashes are in DATASET_MANIFEST.json.

| Split | Original | Retained | Neutral-only removed | Single | Multi | Multi fraction | Mean positives | Mixed neutral retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 43410 | 30587 | 12823 | 24820 | 5767 | 0.188544 | 1.205872 | 1396 |
| validation | 5426 | 3834 | 1592 | 3122 | 712 | 0.185707 | 1.203443 | 174 |
| test | 5427 | 3821 | 1606 | 3154 | 667 | 0.174562 | 1.188694 | 181 |

| Emotion | New ID | Canonical ID | Train support | Validation support | Test support |
|---|---:|---:|---:|---:|---:|
| admiration | 0 | 0 | 4130 | 488 | 504 |
| amusement | 1 | 1 | 2328 | 303 | 264 |
| anger | 2 | 2 | 1567 | 195 | 198 |
| annoyance | 3 | 3 | 2470 | 303 | 320 |
| approval | 4 | 4 | 2939 | 397 | 351 |
| caring | 5 | 5 | 1087 | 153 | 135 |
| confusion | 6 | 6 | 1368 | 152 | 153 |
| curiosity | 7 | 7 | 2191 | 248 | 284 |
| desire | 8 | 8 | 641 | 77 | 83 |
| disappointment | 9 | 9 | 1269 | 163 | 151 |
| disapproval | 10 | 10 | 2022 | 292 | 267 |
| disgust | 11 | 11 | 793 | 97 | 123 |
| embarrassment | 12 | 12 | 303 | 35 | 37 |
| excitement | 13 | 13 | 853 | 96 | 103 |
| fear | 14 | 14 | 596 | 90 | 78 |
| gratitude | 15 | 15 | 2662 | 358 | 352 |
| grief | 16 | 16 | 77 | 13 | 6 |
| joy | 17 | 17 | 1452 | 172 | 161 |
| love | 18 | 18 | 2086 | 252 | 238 |
| nervousness | 19 | 19 | 164 | 21 | 23 |
| optimism | 20 | 20 | 1581 | 209 | 186 |
| pride | 21 | 21 | 111 | 15 | 16 |
| realization | 22 | 22 | 1110 | 127 | 145 |
| relief | 23 | 23 | 153 | 18 | 11 |
| remorse | 24 | 24 | 545 | 68 | 56 |
| sadness | 25 | 25 | 1326 | 143 | 156 |
| surprise | 26 | 26 | 1060 | 129 | 141 |

Canonical IDs are unique within and disjoint across splits. Retained text and labels match the source exactly after neutral removal.
All retained targets are 27-dimensional multi-hot vectors with >=1 positive. Multi-label examples remain multi-label.
Existing duplicate text, if any, is inherited from official splits; no deduplication or resplitting.

```json
{
  "train_validation": {
    "original_exact_text_overlap": 41,
    "retained_exact_text_overlap": 32
  },
  "train_test": {
    "original_exact_text_overlap": 32,
    "retained_exact_text_overlap": 24
  },
  "validation_test": {
    "original_exact_text_overlap": 10,
    "retained_exact_text_overlap": 7
  }
}
```

This is neither the original HypEmo single-label protocol nor the earlier full 28-label protocol.
