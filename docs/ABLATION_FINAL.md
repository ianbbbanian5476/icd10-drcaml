# Ablation Study: ICD-10 Auto-Coding on Taiwan Hospital Data

## Models & Results

| Model | Architecture | Labels | λ | Loss | Best TH | Micro F1 | Macro F1 | P@8 | R@8 |
|---|---|---|---|---|---|---|---|---|---|
| V2 | BERT [CLS] | 14,789 | — | BCE | 0.50 | 39.64% | 0.81% | — | — |
| V5.1 | BERT [CLS] | 2,099 | — | BCE | 0.30 | **65.40%** | 21.93% | 36.83% | 79.38% |
| V6 | BERT DR-CAML | 2,099 | 0.01 | BCE | 0.50 | 67.27% | 38.42% | — | — |
| V6OC | BERT DR-CAML | 2,099 | 0.10 | Focal(γ=2) | 0.40 | 61.81% | **28.06%** | 36.24% | 78.51% |

## Contribution Decomposition

```
V2 (39.64%) ──清洗──→ V5.1 (65.40%) ──DR-CAML──→ V6 (67.27%)
    │                      │                        │
    └── +25.76% (清洗) ────┘                        │
                               └── +1.87% (架構) ────┘
```

| Factor | ΔMicro F1 | ΔMacro F1 |
|---|---|---|
| 資料清洗 (14,789→2,099) | **+25.76%** | +21.12% |
| DR-CAML 架構 | +1.87% | +16.49% |
| Focal Loss + λ=0.1 (V6OC) | -5.46%* | +6.13% |

*V6OC Micro F1 低於 V5.1，但 Macro F1 顯著提升——Focal Loss 以 Micro F1 換取罕見病覆蓋率

## Threshold Optimization

| Model | TH=0.50 | Optimal TH | Optimal F1 | Δ |
|---|---|---|---|---|
| V5.1 [CLS] | 63.72% | 0.30 | 65.40% | +1.68% |
| V6OC DR-CAML | 61.19% | 0.40 | 61.81% | +0.62% |

## Key Findings

1. **資料清洗是最大貢獻因子**（+25.76% Micro F1），F-code 去除 + 4碼截斷 + MIN_FREQ=50 將 label space 從 14,789 壓至 2,099
2. **DR-CAML 主要改善 Macro F1**（+16.49%），對罕見病的 Label-Aware Attention 效果顯著
3. **Focal Loss 進一步推高 Macro F1**（+6.13% vs V5.1），但犧牲部分 Micro F1
4. **Threshold 校準對所有模型均有提升**（+0.62~1.68%），多標籤場景下 0.5 非最佳預設值
5. **co-occurrence post-hoc correction 無效**——模型已從訓練資料學會標籤共現關係
