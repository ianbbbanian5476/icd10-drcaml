# ICD-10 自動編碼：基於 DR-CAML 與語意補償機制的極端多標籤分類

> 台灣台中榮民總醫院臨床病歷訓練 · MIMIC-IV 驗證  
> 2025-2026

---

## 概述

基於 **DR-CAML**（Description Regularized Convolutional Attention for Multi-Label Classification）架構，從出院摘要自動預測 ICD-10 疾病代碼。核心貢獻：

- **東海標準清洗法**：F-code 去除 + 四碼截斷 + MIN_FREQ 過濾，將標籤空間從 14,789 降至 2,099
- **Focal Loss + Description Regularization**：λ=0.1 搭配 Focal Loss（γ=2），Macro F1 較 BCE baseline 提升 6%
- **Threshold 校準**：證明多標籤場景下 threshold 0.3-0.4 優於預設 0.5
- **Domain Gap 量化**：MIMIC-IV 與台灣病歷的文本長度差異（中位數 3,207 vs 422 tokens）

## 主要結果

| 模型 | 資料集 | Micro F1 | Macro F1 | 備註 |
|---|---|---|---|---|
| V5.1 | 台中榮總 | 65.40% | 21.93% | BERT [CLS]，最優 TH=0.3 |
| V6 | 台中榮總 | **67.27%** | **38.42%** | BERT DR-CAML，λ=0.01 |
| V6OC | 台中榮總 | 61.81% | 28.06% | DR-CAML + Focal Loss + λ=0.1，最優 TH=0.4 |
| V6.4 | MIMIC-IV | 41.44% | 7.03% | DR-CAML on 清洗後 MIMIC |

## 資料管線

```
原始病歷 (2015-2022)
    │
    ▼ V2: 14,789 labels (MIN_FREQ=1，無過濾)
    │
    ▼ V4: 14,219 labels (去除 F-code)
    │
    ▼ V5: 2,099 labels (四碼截斷 + MIN_FREQ=50)
    │
    ▼ V6: DR-CAML 訓練於 V5 資料
```

## 架構：DR-CAML

```
臨床病歷 ──→ Bio_ClinicalBERT ──→ H ∈ R^(seq×768)
                                        │
                              Label-Aware Attention
                                        │
                              V ∈ R^(num_labels×768)
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                 out_weights · V              ||out_weights - target||²
                 → BCE / Focal Loss           → MSE Regularization (λ)
```

- **Label-Aware Attention**：每個 ICD 代碼學會關注病歷的不同段落
- **Description Regularization**：分類器權重被拉向 ICD 代碼官方描述的 Bio_ClinicalBERT embedding
- **Focal Loss**（γ=2）：抑制易分類負樣本的 loss，強迫模型關注罕見疾病

## 重現步驟

### 環境需求

- Python 3.10+
- PyTorch 2.x、Transformers、scikit-learn
- GPU：12GB+ VRAM（於 RTX 8000-12Q 測試）

### 訓練

```bash
cd src
python train_v6oc.py  # DR-CAML + Focal Loss，2,099 labels
```

### 評估

```bash
python universal_evaluator.py \
    --model_type drcaml \
    --model_weights v6oc_best_model.pth \
    --label_map label_map_v5.json \
    --val_data val_v5.json \
    --embeddings v5_label_embeddings.pt
```

## 目錄結構

```
├── src/                         # 訓練與評估腳本
│   ├── train_v6oc.py            # DR-CAML + Focal Loss
│   ├── train_v6oc_1024.py       # 1024-token 版本
│   ├── train_v5_1_bert_cls.py   # BERT [CLS] baseline
│   ├── train_v3_longformer.py   # Longformer 實驗
│   ├── train_v6_4_drcaml_mimic.py  # MIMIC-IV 版本
│   ├── universal_evaluator.py   # 通用評估器
│   ├── build_mf20_data.py       # MF20 資料建構
│   ├── build_v5_cleaned.py      # V5 清洗管線
│   └── step7_prep_mimic_iv.py   # MIMIC-IV 前處理
├── docs/
│   ├── ABLATION_FINAL.md        # Ablation study
│   └── AUDIT_REPORT.md          # 實驗審計報告
├── results/
│   └── experiment_tracker.csv   # 實驗追蹤
└── README.md
```

## 參考文獻

- Mullenbach et al. (2018). "Explainable Prediction of Medical Codes from Clinical Text." *NAACL*.
- Lin et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.
- Li & Yu (2020). "Multi-Filter Residual CNN for Text Classification."
- Vu et al. (2020). "A Label Attention Model for ICD Coding." *IJCAI*.
- PLM-ICD (2022). "Automatic ICD Coding with Pretrained Language Models." *ClinicalNLP*.

## 授權

MIT
