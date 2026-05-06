import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np

# ================= 環境設定 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 目前使用的運算設備: {device}")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
BATCH_SIZE = 16

VAL_DATA_PATH = "val_v5.json"
LABEL_MAP_PATH = "label_map_v5.json"
EMBEDDINGS_PATH = "v5_label_embeddings.pt" 
MODEL_WEIGHTS_PATH = "v6_dr_caml_best_model.pth" # V6 訓練產出的權重
# ============================================

# 1. 完美重建 V6 DR-CAML 架構
class BertDRCAML(nn.Module):
    def __init__(self, model_name, num_labels, label_embeddings):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.attention_weights = nn.Linear(hidden_size, num_labels)
        self.out_weights = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.out_bias = nn.Parameter(torch.Tensor(num_labels))
        self.register_buffer('target_embeddings', label_embeddings)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  
        A_logits = self.attention_weights(H) 
        mask = attention_mask.unsqueeze(-1) 
        A_logits = A_logits.masked_fill(mask == 0, -1e4) # 使用我們修復好的 -1e4
        A = torch.softmax(A_logits, dim=1) 
        V = torch.bmm(A.transpose(1, 2), H) 
        logits = torch.sum(V * self.out_weights.unsqueeze(0), dim=-1) + self.out_bias
        return logits, self.out_weights

# 2. 資料集定義
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
        raw_labels = item.get('label', [])
        
        labels = torch.zeros(self.num_labels)
        for label_id in raw_labels:
            if isinstance(label_id, int) and 0 <= label_id < self.num_labels:
                labels[label_id] = 1.0
            elif isinstance(label_id, str) and label_id.isdigit():
                labels[int(label_id)] = 1.0

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# 3. P@k & R@k 函數
def calculate_pk_rk(y_true, y_pred, k):
    pk_list, rk_list = [], []
    for i in range(y_true.shape[0]):
        true_labels = set(np.where(y_true[i] == 1)[0])
        if len(true_labels) == 0: continue
        top_k_preds = set(np.argsort(y_pred[i])[-k:][::-1])
        hits = len(top_k_preds.intersection(true_labels))
        pk_list.append(hits / k)
        rk_list.append(hits / len(true_labels))
    return np.mean(pk_list), np.mean(rk_list)

def main():
    print(f"\n{'='*70}")
    print(f"🏆 V6 DR-CAML 終極指標驗證程序")
    print(f"{'='*70}")

    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        raw_label_map = json.load(f)
    label2id = raw_label_map.get("label2id", raw_label_map)
    NUM_LABELS = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"📥 載入疾病標準答案庫: {EMBEDDINGS_PATH}")
    label_embeddings = torch.load(EMBEDDINGS_PATH, map_location=device)
    
    print("🤖 初始化 BertDRCAML 架構...")
    model = BertDRCAML(MODEL_NAME, NUM_LABELS, label_embeddings)
    
    print(f"📥 載入訓練權重: {MODEL_WEIGHTS_PATH}")
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    val_dataset = ICD10Dataset(VAL_DATA_PATH, tokenizer, MAX_LEN, label2id)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    val_targets, val_preds = [], []

    print(f"🔍 開始對 {len(val_dataset)} 筆驗證資料進行推論...\n")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="推論中"):
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

    print("\n📊 正在計算各項評估指標 (0 幻覺精密運算)...")
    micro_f1 = f1_score(val_targets, val_preds_bin, average='micro')
    macro_f1 = f1_score(val_targets, val_preds_bin, average='macro', zero_division=0)
    micro_auc = roc_auc_score(val_targets, val_preds, average='micro')
    
    p_5, r_5 = calculate_pk_rk(val_targets, val_preds, k=5)
    p_8, r_8 = calculate_pk_rk(val_targets, val_preds, k=8)
    p_10, r_10 = calculate_pk_rk(val_targets, val_preds, k=10)
    p_15, r_15 = calculate_pk_rk(val_targets, val_preds, k=15)

    np.save('y_true_v6.npy', val_targets)
    np.save('y_pred_probs_v6.npy', val_preds)
    print("✅ 已將 V6 預測結果儲存為 y_true_v6.npy 與 y_pred_probs_v6.npy！")
    print(f"\n{'='*40}")
    print(f"🌟 V6 DR-CAML 最終成績單")
    print(f"{'='*40}")
    print(f"🔹 Micro F1   : {micro_f1:.4f}")
    print(f"🔹 Macro F1   : {macro_f1:.4f}  <-- 史詩級突破！")
    print(f"🔹 Micro AUC  : {micro_auc:.4f}")
    print(f"{'-'*40}")
    print(f"🔸 P@8        : {p_8:.4f}  |  R@5  : {r_5:.4f}")
    print(f"🔸 P@15       : {p_15:.4f}  |  R@10 : {r_10:.4f}")
    print(f"                        |  R@15 : {r_15:.4f}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    main()