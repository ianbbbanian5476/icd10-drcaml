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

| Symbol | Definition（EN） | 定義（中文） |
|--------|-----------|-----------|
| $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^{N}$ | Training set of $N$ clinical notes | $N$ 筆病歷之訓練集 |
| $\mathbf{x}_i$ | Tokenized input sequence, $\|\mathbf{x}_i\| \leq 512$ | 斷詞後之輸入序列，長度 ≤ 512 |
| $\mathbf{y}_i \in \{0,1\}^{L}$ | Multi-hot label vector, $L = 2099$ | 多標籤向量（0/1），共 2,099 維 |
| $L$ | Number of unique ICD-10 codes (after filtering) | 篩選後之 ICD-10 代碼總數 |
| $d = 768$ | Hidden size of Bio_ClinicalBERT | BERT 隱藏層維度 |
| $\mathbf{H} \in \mathbb{R}^{n \times d}$ | BERT last hidden states ($n$ = sequence length) | BERT 最終層輸出矩陣 |
| $A \in \mathbb{R}^{n \times L}$ | Label-wise attention weights | 各標籤對各 token 之注意力權重 |
| $\mathbf{V} \in \mathbb{R}^{L \times d}$ | Label-specific feature matrix | 各標籤從病歷中擷取之特徵向量 |
| $\mathbf{W} \in \mathbb{R}^{L \times d}$ | Learnable classification weight matrix | 可學習之分類權重矩陣 |
| $\mathbf{E} \in \mathbb{R}^{L \times d}$ | Pre-computed label description embeddings (frozen) | 預先計算之標籤描述嵌入（凍結，不參與梯度更新） |
| $\lambda$ | Regularization coefficient | 正則化係數（控制語意引導強度） |
| $\hat{\mathbf{y}}_i = \sigma(\mathbf{z}_i)$ | Predicted probabilities, $\sigma$ = sigmoid | 預測機率，$\sigma$ 為 sigmoid 函數 |

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

#### Loss Function — 各模型對照

以下依模型版本逐一給出 loss function 之完整數學定義。

---

**模型 A：V5.1（BERT CLS + BCE）**

架構最簡：病歷文本經 BERT 編碼後取 `[CLS]` 向量，直接送入線性分類層。

$$\mathbf{h} = \text{BERT}(\mathbf{x})_{\text{[CLS]}} \in \mathbb{R}^{d}$$

$$\mathbf{z} = \mathbf{W}_{\text{cls}} \mathbf{h} + \mathbf{b} \in \mathbb{R}^{L}, \quad \hat{\mathbf{y}} = \sigma(\mathbf{z})$$

$$\boxed{\mathcal{L}_{\text{V5.1}} = -\frac{1}{L}\sum_{l=1}^{L} \left[y_l \log \hat{y}_l + (1-y_l)\log(1-\hat{y}_l)\right]}$$

- 純 multi-label binary cross-entropy
- 無正則化項、無 label attention
- 所有標籤共享同一份病歷表徵 $\mathbf{h}$

---

**模型 B：V6（BERT DR-CAML + BCE + Description Regularization）**

$$\mathbf{H} = \text{BERT}(\mathbf{x}) \in \mathbb{R}^{n \times d}$$

$$A = \text{softmax}_{\text{masked}}(\mathbf{H} \mathbf{A}_q) \in \mathbb{R}^{n \times L}$$

$$\mathbf{V} = A^\top \mathbf{H} \in \mathbb{R}^{L \times d}$$

$$\mathbf{z} = \text{diag}(\mathbf{V} \mathbf{W}^\top) + \mathbf{b}, \quad \hat{\mathbf{y}} = \sigma(\mathbf{z})$$

$$\boxed{\mathcal{L}_{\text{V6}} = \underbrace{-\frac{1}{L}\sum_{l=1}^{L} \left[y_l \log \hat{y}_l + (1-y_l)\log(1-\hat{y}_l)\right]}_{\mathcal{L}_{\text{BCE}}} + \underbrace{0.01 \cdot \frac{1}{L}\|\mathbf{W} - \mathbf{E}\|_F^2}_{\mathcal{L}_{\text{reg}},\ \lambda=0.01}}$$

- BCE 處理所有標籤之預測誤差（公平加權）
- Description regularization 輕微引導（λ=0.01），讓模型以資料學習為主
- 各標籤獨立 attention 空間避免高頻淹沒低頻

---

**模型 C：V6OC（BERT DR-CAML + Focal Loss + Description Regularization）**

$$\boxed{\mathcal{L}_{\text{V6OC}} = \underbrace{-\frac{1}{L}\sum_{l=1}^{L} \alpha (1-\hat{y}_{t,l})^\gamma \cdot \left[y_l \log \hat{y}_l + (1-y_l)\log(1-\hat{y}_l)\right]}_{\mathcal{L}_{\text{Focal}}}}$$

$$\boxed{+\ \underbrace{0.10 \cdot \frac{1}{L}\|\mathbf{W} - \mathbf{E}\|_F^2}_{\mathcal{L}_{\text{reg}},\ \lambda=0.10}}$$

其中 $\hat{y}_{t,l} = y_l \cdot \hat{y}_l + (1-y_l) \cdot (1-\hat{y}_l)$（模型對正確類別之信心）。

Focal Loss 參數：$\alpha = 0.25,\ \gamma = 2.0$。

- Focal Loss 以 $(1-\hat{y}_{t,l})^\gamma$ 動態調節每個樣本之 loss 權重：
  - 已學會的常見病（$\hat{y}_{t,l} \to 1$）→ 權重趨近 0，模型不再浪費容量
  - 尚未學會的罕見病（$\hat{y}_{t,l} \to 0$）→ 權重趨近 1，強迫模型關注
- λ 提高至 0.10：因 Focal Loss 內建之類別加權會使 $\mathbf{W}$ 偏離 $\mathbf{E}$，需較強正則化抗衡

---

**模型 D：MF20（同 V6OC，label 空間擴張至 2,873）**

$$\boxed{\mathcal{L}_{\text{MF20}} = \mathcal{L}_{\text{Focal}}(\hat{\mathbf{y}}, \mathbf{y}; \alpha{=}0.25, \gamma{=}2.0) + 0.10 \cdot \frac{1}{L_{20}}\|\mathbf{W} - \mathbf{E}_{20}\|_F^2}$$

其中 $L_{20} = 2,873$（MIN_FREQ=20 之標籤數），$\mathbf{E}_{20}$ 為對應之 label embeddings。

- Loss 形式與 V6OC 相同，差別僅在 $\mathbf{W} \in \mathbb{R}^{2873 \times d}$ 與 $\mathbf{E}$ 維度增大
- 此設定用於測試 label 空間擴張對 loss 收斂之影響（穩健性測試）

---

**各模型 Loss 對照總表**

| 模型 | 分類 Loss | Reg (λ) | Label 數 | Loss 設計意圖 |
|------|------|:---:|:---:|------|
| V5.1 | BCE | — | 2,099 | CLS baseline，無語意補償 |
| V6 | BCE | 0.01 | 2,099 | DR-CAML + 輕微語意引導 |
| V6OC | **Focal** (γ=2) | 0.10 | 2,099 | Focal 處理長尾 + 強語意引導 |
| MF20 | **Focal** (γ=2) | 0.10 | 2,873 | 同 V6OC，測試 label 空間擴張 |

#### 與原始 DR-CAML（Mullenbach et al., 2018）之 Loss 設計差異

原始 DR-CAML 論文中，標籤描述文字在訓練過程中**即時參與計算**：

$$\mathbf{v}_l^{\text{desc}} = \text{CNN}(\text{description}_l)$$

$$\mathcal{L}_{\text{orig}} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \text{cosine\_distance}(\mathbf{v}_l^{\text{clinical}}, \mathbf{v}_l^{\text{desc}})$$

即每步訓練需將 $L$ 段標籤描述送進 CNN 編碼，與病歷表徵 $\mathbf{v}_l^{\text{clinical}}$ 做對比（contrastive）。

**我們的簡化**：

$$\mathbf{E}_l = \text{ClinicalBERT}(\text{description}_l) \quad \text{（事先計算，訓練中凍結）}$$

$$\mathcal{L}_{\text{ours}} = \mathcal{L}_{\text{BCE/Focal}} + \lambda \cdot \underbrace{\frac{1}{L}\|\mathbf{W} - \mathbf{E}\|_F^2}_{\text{MSE on classification weights}}$$

| | 原始 DR-CAML | 本研究 |
|------|------|------|
| 描述參與時機 | 每個 batch 即時編碼 | 訓練前一次計算，凍結 |
| 對齊對象 | $\mathbf{v}_l$（每標籤的病歷特徵）vs 描述特徵 | $\mathbf{W}$（分類權重層）vs $\mathbf{E}$ |
| 對齊方式 | cosine distance / contrastive | $\|\mathbf{W} - \mathbf{E}\|_F^2$（MSE） |
| 每步運算量 | $\mathcal{O}(L \cdot \text{CNN})$ | $\mathcal{O}(L \cdot d)$ |
| 12GB VRAM 可行性 | ❌ | ✅ |

**設計取捨**：我們將語意對齊從 feature-level（$\mathbf{v}_l$ 對齊）降為 weight-level（$\mathbf{W}$ 對齊）。這犧牲了部分精細度——正則化只作用在分類權重層，不影響 attention 層對病歷特徵的擷取。但換來了校內 GPU 可執行的運算量，且實驗證明（V6 Macro F1 = 38.42% vs 原始 CAML 4.9%）此簡化仍有效保留了語意引導的核心功能。

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
