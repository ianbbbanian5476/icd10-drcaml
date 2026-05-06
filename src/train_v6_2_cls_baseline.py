import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np

# ================= 1. vGPU 相容性與效能設定 =================
print("[系統] 套用 AICS-ST05-U13 vGPU 相容性設定...")
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 目前使用的運算設備: {device}")

# ================= 2. 超參數設定 =================
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
BATCH_SIZE = 16      # Baseline 模型沒有 Attention 矩陣，顯存通常更省，維持 16 沒問題
EPOCHS = 10
LEARNING_RATE = 3e-5

# 讀取 MIMIC V6.4 (東海標準版) 資料與字典
TRAIN_DATA_PATH = "train_v6_4_mimic_clean.json"
LABEL_MAP_PATH = "label_map_v6_4.json"
BEST_MODEL_PATH = "mimic_v6_2_cls_best_model.pth" # 基準線模型存檔名稱
# ==============================================

# ================= 3. 讀取與動態校正 Label Map =================
print(f"📂 正在載入標籤字典...")
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    raw_label_map = json.load(f)

label2id = raw_label_map.get("label2id", raw_label_map)
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
        
        # 使用我們洗出來的欄位
        raw_labels = item.get('all_codes', []) 
        
        labels = torch.zeros(self.num_labels)
        # 透過字典配對標籤 ID
        for code in raw_labels:
            if code in self.label2id:
                label_idx = self.label2id[code]
                labels[label_idx] = 1.0

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# ================= 5. 主訓練流程 =================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 🤖 初始化模型：純 CLS 的 Baseline 模型 (AutoModelForSequenceClassification)
    print(f"🤖 初始化模型：純 CLS Baseline (HuggingFace AutoModelForSequenceClassification)")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS, 
        problem_type="multi_label_classification"
    )
    model.to(device)

    # 載入並切分資料集
    full_dataset = ICD10Dataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN, label2id)    
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"✅ 資料集切分完成: 訓練集 {len(train_dataset)} 筆, 驗證集 {len(val_dataset)} 筆")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler()

    print(f"\n⚔️ V6.2 純 CLS 基準線訓練正式啟動")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(Loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 訓練完畢 | Avg Loss: {avg_loss:.4f}")
        
        # --- 驗證階段 ---
        model.eval()
        val_targets, val_preds = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.sigmoid(logits)
                
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(probs.cpu().numpy())

        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        val_preds_bin = (val_preds > 0.5).astype(int)

        micro_f1 = f1_score(val_targets, val_preds_bin, average='micro')
        macro_f1 = f1_score(val_targets, val_preds_bin, average='macro', zero_division=0)
        micro_auc = roc_auc_score(val_targets, val_preds, average='micro')
        
        print(f"🎯 驗證集表現 -> Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f} | Micro AUC: {micro_auc:.4f}\n")

    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"✅ V6.2 基準線訓練完成！已儲存至 {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()