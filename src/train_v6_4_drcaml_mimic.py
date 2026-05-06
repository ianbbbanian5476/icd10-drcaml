import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
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
BATCH_SIZE = 16      # 您的伺服器 (RTX8000 48GB) 性能強大，我們試著拉高到 16 加速訓練，若 OOM 再改回 8
EPOCHS = 10
LEARNING_RATE = 3e-5
LAMBDA_REG = 0.01    # DR-CAML 論文中的正規化權重 (控制標準答案的引導力道)

# 🚀 [修改點]: 讀取剛洗好的 MIMIC V6.4 (東海標準版) 資料
TRAIN_DATA_PATH = "train_v6_4_mimic_clean.json"
# 🚨 [注意]: 我們剛剛的前處理腳本沒有幫您切 Val，這裡需要改成讀同一份，
# 或等一下我們在程式碼裡面加一行自動切分的邏輯。我們先設定為 None。
VAL_DATA_PATH = None 
LABEL_MAP_PATH = "label_map_v6_4.json"
EMBEDDINGS_PATH = "mimic_v6_4_embeddings.pt" # 剛剛產生的 MIMIC 專屬向量庫

# 🚀 [修改點]: 防止覆蓋舊的榮總模型，給它一個新名字
BEST_MODEL_PATH = "mimic_v6_4_best_model.pth"
# ==============================================

# ================= 3. 讀取與動態校正 Label Map =================
print(f"📂 正在載入 V5 標籤字典...")
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    raw_label_map = json.load(f)

label2id = raw_label_map.get("label2id", raw_label_map)
NUM_LABELS = len(label2id)
print(f"✅ 成功載入字典，實際包含 {NUM_LABELS} 個標籤！")

# ================= 4. 神經網路定義：DR-CAML 架構 =================
class BertDRCAML(nn.Module):
    def __init__(self, model_name, num_labels, label_embeddings):
        super().__init__()
        # 底層特徵萃取 (Bio_ClinicalBERT)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Label-Aware Attention 機制 (2099 個專屬調查員)
        self.attention_weights = nn.Linear(hidden_size, num_labels)
        
        # 最終分類器權重
        # ⚠️ 注意：這裡的 out_weights 就是我們要用來跟「標準答案」算距離的變數！
        self.out_weights = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.out_bias = nn.Parameter(torch.Tensor(num_labels))
        nn.init.xavier_uniform_(self.out_weights)
        nn.init.zeros_(self.out_bias)
        
        # 載入疾病標準答案向量 (不需要參與梯度更新，它是個定海神針)
        self.register_buffer('target_embeddings', label_embeddings)

    def forward(self, input_ids, attention_mask):
        # 1. BERT 萃取所有字元的特徵 H
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # [Batch, Seq_Len, Hidden_Size]

        # 2. 計算每個疾病的專屬注意力權重 A
        A_logits = self.attention_weights(H) # [Batch, Seq_Len, Num_Labels]
        mask = attention_mask.unsqueeze(-1)
        A_logits = A_logits.masked_fill(mask == 0, -1e4)
        A = torch.softmax(A_logits, dim=1) 

        # 3. 加權融合，得到每個疾病對應這份病歷的專屬特徵 V
        V = torch.bmm(A.transpose(1, 2), H) # [Batch, Num_Labels, Hidden_Size]

        # 4. 進行最終預測 (Logits)
        logits = torch.sum(V * self.out_weights.unsqueeze(0), dim=-1) + self.out_bias
        
        return logits, self.out_weights

# ================= 5. 資料集定義 (ICD10Dataset) =================
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
        
        # 1. 確保抓取我們 V6.4 洗出來的欄位
        raw_labels = item.get('all_codes', []) 
        
        labels = torch.zeros(self.num_labels)
        
        # 2. 透過 label2id 字典，將 "I10" 等字串轉換為正確的向量索引
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

# ================= 6. 主訓練流程 =================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 載入階段一算好的 Embeddings 作為標準答案
    print(f"📥 載入疾病標準答案庫: {EMBEDDINGS_PATH}")
    label_embeddings = torch.load(EMBEDDINGS_PATH, map_location=device)
    
    # 初始化 DR-CAML 模型
    print(f"🤖 初始化模型：BertDRCAML (包含 Label-Aware Attention 與 Description Regularization)")
    model = BertDRCAML(MODEL_NAME, NUM_LABELS, label_embeddings)
    model.to(device)

    full_dataset = ICD10Dataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN, label2id)    
    
    # 按照 90% 訓練、10% 驗證的比例切分
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

    # 定義雙重 Loss 函數
    bce_criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss() # 用來計算 L2 距離
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n⚔️ V6 DR-CAML 終極訓練正式啟動")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_bce, total_reg = 0, 0, 0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # 模型除了吐出預測機率，還會把當下學到的「分類權重」交出來
                logits, current_weights = model(input_ids, attention_mask)
                
                # 1. 傳統的分類 Loss (BCE)
                loss_bce = bce_criterion(logits, labels)
                
                # 2. DR-CAML 獨有的正規化 Loss (MSE 距離)
                # 比較「模型目前的權重」與「標準答案」的差距
                loss_reg = mse_criterion(current_weights, model.target_embeddings)
                
                # 3. 結合總 Loss
                loss = loss_bce + LAMBDA_REG * loss_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_reg += loss_reg.item()
            
            # 在進度條顯示兩種 Loss 的比例，讓您能隨時監控
            loop.set_postfix(
                BCE=f"{loss_bce.item():.4f}", 
                REG=f"{(LAMBDA_REG * loss_reg).item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 訓練完畢 | Avg Total Loss: {avg_loss:.4f} (BCE: {total_bce/len(train_loader):.4f}, REG: {(LAMBDA_REG * total_reg)/len(train_loader):.4f})")
        
        # --- 驗證階段 ---
        model.eval()
        val_targets, val_preds = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                with torch.cuda.amp.autocast():
                    logits, _ = model(input_ids, attention_mask)
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

    torch.save(model.state_dict(), "v6_dr_caml_best_model.pth")
    print("✅ V6 DR-CAML 訓練完成，這將是您突破 Macro F1 瓶頸的歷史性時刻！")

if __name__ == "__main__":
    main()