import os

# ================= 1. AICS-ST05-U13 vGPU 相容性設定 =================
print("[系統] 套用 AICS-ST05-U13 vGPU 相容性與效能設定...")
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"
# ⚠️ 絕對禁止加入 PYTORCH_NO_CUDA_MEMORY_CACHING，否則會導致嚴重的降速與 OOM
# ====================================================================

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np

# 確認運算設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 目前使用的運算設備: {device}")

# ================= 2. 超參數設定 (V2 架構 + V5 資料) =================
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
BATCH_SIZE = 8     # 12GB VRAM 建議從 8 開始，若訓練穩定且顯存有餘裕再調升至 16
EPOCHS = 10
LEARNING_RATE = 3e-5

TRAIN_DATA_PATH = "train_v5.json"
VAL_DATA_PATH = "val_v5.json"
LABEL_MAP_PATH = "label_map_v5.json"
# ====================================================================

# ================= 3. 讀取與動態校正 Label Map =================
print(f"📂 正在載入 V5 標籤字典...")
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    raw_label_map = json.load(f)

# 防呆機制：自動判斷字典是單層還是雙層結構，避免 [8, 2] 的維度災難
if "label2id" in raw_label_map:
    label2id = raw_label_map["label2id"]
else:
    label2id = raw_label_map

NUM_LABELS = len(label2id)
print(f"✅ 成功載入字典，實際包含 {NUM_LABELS} 個標籤！")

# ================= 4. 資料集定義 (ICD10Dataset) =================
class ICD10Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, label2id):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.num_labels = len(label2id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        
        # --- 🚀 關鍵修正：直接讀取已經轉好 ID 的 label 陣列 ---
        raw_labels = item.get('label', [])
        
        # 建立 One-Hot 標籤張量
        labels = torch.zeros(self.num_labels)
        for label_id in raw_labels:
            # 確保 ID 沒有超出範圍，並且是整數
            if isinstance(label_id, int) and 0 <= label_id < self.num_labels:
                labels[label_id] = 1.0
            elif isinstance(label_id, str) and label_id.isdigit():
                # 預防萬一它是字串型態的數字 "226"
                labels[int(label_id)] = 1.0
        # --------------------------------------------------------

        # BERT 標準右截斷
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# ================= 5. 主訓練流程 =================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 初始化傳統的 [CLS] 分類頭模型
    print(f"🤖 初始化模型：AutoModelForSequenceClassification (輸出維度: {NUM_LABELS})")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification" # 自動套用 BCEWithLogitsLoss
    )
    model.to(device)

    # 準備 DataLoader
    train_dataset = ICD10Dataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN, label2id)
    val_dataset = ICD10Dataset(VAL_DATA_PATH, tokenizer, MAX_LEN, label2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # 設定優化器與學習率排程
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    # 啟用 AMP 自動混合精度，適應 RTX 8000 記憶體並加速訓練
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n⚔️ V5 BERT 基準測試正式啟動 (純淨資料 + 傳統 CLS 架構)")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            # AMP 前向傳播
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            # AMP 反向傳播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} 訓練平均 Loss: {total_loss/len(train_loader):.4f}")
        
        # --- 驗證階段 ---
        model.eval()
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.sigmoid(logits) # 將 logits 轉換為 0~1 的機率
                
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(probs.cpu().numpy())

        # 計算評估指標
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        val_preds_bin = (val_preds > 0.5).astype(int)

        micro_f1 = f1_score(val_targets, val_preds_bin, average='micro')
        macro_f1 = f1_score(val_targets, val_preds_bin, average='macro', zero_division=0)
        micro_auc = roc_auc_score(val_targets, val_preds, average='micro')
        
        print(f"🎯 驗證集表現 -> Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f} | Micro AUC: {micro_auc:.4f}\n")

    # 儲存最終模型權重
    torch.save(model.state_dict(), "v5_bert_cls_best_model.pth")
    print("✅ V5 BERT (CLS 版) 訓練完成，權重已成功儲存。")

if __name__ == "__main__":
    main()