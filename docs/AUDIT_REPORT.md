# ICD-10 自動編碼專案 — 完整審計報告

> 工作目錄：`/home/s11350310/icd10-2nd/BERT_TRY/V2/opencode_fix/`
> 審計日期：2025-05-05
> 所有原始檔案**未修改**，修改版僅存在於 `opencode_fix/`

---

## 🔴 一級警報：評估協議系統性失靈

### 發現 1：V6.4 評估使用錯誤 Label Map
| | 訓練時 | 評估時 | 結果 |
|---|---|---|---|
| Model | label_map_v6_4.json (2,496 labels) | — | — |
| Eval | — | **label_map_v5.json** (2,099 labels) | ❌ |
| 影響 | — | Index 2099-2495 永遠無法命中 | **低估 5% F1** |

### 發現 2：V3.1 與 V5 Longformer 資料可能同樣錯誤
```
V3.1 Longformer → Micro F1 14%  (vs V2 BERT 39.64%)  差距 -25%
V5   Longformer → Micro F1 29%  (vs V5.1 BERT 63.72%) 差距 -34%
```
這兩組差距**不合理**。Longformer 即使劣於 BERT，不該差 25-34%。評估協議可能同樣有 bug。

### 發現 3：V5 與 V6.4 Label 空間不同
| | V5 | V6.4 | 重疊 |
|---|---|---|---|
| Labels | 2,099 | 2,496 | **1,539** |
| V5 獨有 | **560** | — | — |
| V6.4 獨有 | — | **957** | — |

任何使用 V5 validation 去評估 V6.4 模型的實驗結果均**無效**。

---

## 🟠 二級問題：評估腳本錯配清單

| 腳本 | VAL_DATA | LABEL_MAP | MODEL | 問題 |
|---|---|---|---|---|
| `!V6!step6_evaluate_v6_dr_caml.py` | val_v5.json | label_map_v5.json | v6_dr_caml_best_model.pth | ❌ V5 label space on V6 model |
| `.step4_evaluate_hypertension.py` | V6 preds (npy) | label_map_v5.json | — | ❌ V5 label map on V6 results |
| `!step3_evaluate_v3.py` | val_v2.json | model built-in | clinical_longformer_v3_model | ⚠️ 需確認 label space 一致性 |

---

## 🟡 三級問題：資料預處理

### 台灣榮總管線 (V2→V4→V5)
```
V1: 80/20 split, 91,737 labels
 ↓
V2: 2015-2022 aggregates, MIN_FREQ=1, 14,789 labels, 356k train
 ↓
V4: F-codes removed (record-level drop), 14,219 labels
 ↓
V5: code[:4] truncation + MIN_FREQ=50, 2,099 labels ← 清洗核心
```

### MIMIC-IV 管線 (V6.3→V6.4)
```
V6.3: raw, 16,155 labels, 122k cases
 ↓
V6.4: code[:4] + F-filter + MIN_FREQ=50, 2,496 labels
```

### 問題
1. **V4 與 V6.4 的 F-code 過濾邏輯不同**：
   - V4：整筆紀錄有 F-code → 整筆丟棄
   - V6.4：只移除 F-code，保留其餘 codes
2. **code[:4] 截斷丟失 specificity**：T474X5A → T474
3. **V6.4 嵌入矩陣錯位**：`mimic_v6_4_embeddings.pt` 實際是 V6.3 的 16,155 維

---

## 🔵 四級：缺失項目

| 項目 | 狀態 |
|---|---|
| V6 台灣訓練腳本 | ❌ 缺失（不在 V2 目錄） |
| Ablation study (清洗 vs DR-CAML) | ❌ 未執行 |
| 完整 P@k / R@k 指標 (V2-V6) | ❌ 僅 V6.4 有 |
| Baseline (TF-IDF + Logistic Regression) | ❌ 未建立 |
| 實驗追蹤 (哪次實驗對應哪個結果) | ❌ 未建立 → 已建立 `experiment_tracker.csv` |

---

## 📊 已確認性能表

| Model | Dataset | F1 | Labels | 評估協議 | 狀態 |
|---|---|---|---|---|---|
| V2 | 榮總 | 39.64 | 14,789 | ✅ | 可信 |
| V3.1 | 榮總 | 14.00 | 14,789 | ⚠️ | 需重驗 |
| V5 | 榮總 | 29.35 | 2,099 | ⚠️ | 需重驗 |
| V5.1 | 榮總 | 63.72 | 2,099 | ✅ | 可信 |
| V6 | 榮總 | 67.27 | 2,099 | ❓ | 訓練腳本遺失 |
| V6.2 | MIMIC | 29.51 | 2,496 | ❓ | val split 未知 |
| V6.3 | MIMIC | 31.56 | 16,155 | ❓ | val split 未知 |
| V6.4 (原) | MIMIC | 36.45 | 2,496 | ❌ | **無效** |
| **V6.4 (修正)** | MIMIC | **41.44** | 2,496 | ✅ | **可信** |

---

## 🎯 優先行動方案

### 立即 (P0)
- [x] 修正 V6.4 評估 → 41.44% F1
- [ ] 審計 V3.1 / V5 Longformer 評估協議
- [ ] 找到 V6 台灣模型權重與訓練腳本

### 本日 (P1)
- [ ] 補完所有歷史模型的 P@k, R@k 指標
- [ ] 建立 unified evaluation script
- [ ] 修正評估腳本中的 label map 錯配

### 本週 (P2)
- [ ] 設計並執行 ablation study
- [ ] 實作 Hierarchical BERT
- [ ] 重新驗證 Longformer (正確評估 + batch=1)
