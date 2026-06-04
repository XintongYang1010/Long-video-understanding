# Qwen3-VL MA-EgoQA Subset Summary

| Condition | Agents | Items | Correct | Accuracy | Invalid | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Jack | Jake | 1741 | 448 | 25.73% | 0 | 966.3 |
| Alice | Alice | 1741 | 464 | 26.65% | 0 | 953.2 |
| Katrina | Katrina | 1741 | 482 | 27.69% | 0 | 975.8 |
| Jack_Alice | Jake+Alice | 1741 | 506 | 29.06% | 0 | 1387.1 |
| Jack_Katrina | Jake+Katrina | 1741 | 533 | 30.61% | 0 | 1395.6 |
| Alice_Katrina | Alice+Katrina | 1741 | 558 | 32.05% | 0 | 1385.9 |

Single-agent mean accuracy: 26.69%
Pair-agent mean accuracy: 30.58%
Best pair: Alice_Katrina (32.05%)
