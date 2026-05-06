#!/usr/bin/env python3
"""Re-evaluate V3.1 Longformer (1024 tokens) on V2 data with correct label map."""
import os, json, torch, numpy as np, sys
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

MODEL_DIR = "/home/s11350310/icd10-2nd/BERT_TRY/V2/clinical_longformer_v3_model"
VAL_DATA = "/home/s11350310/icd10-2nd/BERT_TRY/V2/val_v2.json"
LABEL_MAP = "/home/s11350310/icd10-2nd/BERT_TRY/V2/label_map_v2.json"
MAX_LEN = 1024
BATCH_SIZE = 8 if device.type == "cuda" else 1

with open(LABEL_MAP, 'r') as f:
    lm = json.load(f)
label2id = lm.get("label2id", lm)
NUM_LABELS = len(label2id)
print(f"V2 Labels: {NUM_LABELS}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

with open(VAL_DATA, 'r') as f:
    val_data = json.load(f)
print(f"Val samples: {len(val_data)}", flush=True)

class ValDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        raw_labels = item.get('label', [])
        labels = torch.zeros(NUM_LABELS)
        for lid in raw_labels:
            if isinstance(lid, int) and 0 <= lid < NUM_LABELS:
                labels[lid] = 1.0
        enc = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(), 'labels': labels}

ds = ValDataset(val_data, tokenizer, MAX_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4 if device.type=="cuda" else 0, pin_memory=True)

val_targets, val_preds = [], []
for batch in tqdm(dl, desc="V3.1 eval"):
    logits = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits
    val_targets.extend(batch['labels'].numpy())
    val_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

val_targets = np.array(val_targets); val_preds = np.array(val_preds)

print("\n=== THRESHOLD SWEEP ===", flush=True)
for th in [0.5, 0.3, 0.2, 0.1, 0.05]:
    y_bin = (val_preds > th).astype(int)
    tp = (val_targets * y_bin).sum(); fp = ((1-val_targets) * y_bin).sum(); fn = (val_targets * (1-y_bin)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0; rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    auc = roc_auc_score(val_targets, val_preds, average='micro')
    active = (y_bin.sum(axis=0) > 0).sum()
    print(f"  TH={th:.2f}: Micro F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} AUC={auc:.4f} active={active}/{NUM_LABELS}", flush=True)

def pk_rk(y_true, y_pred, k):
    pk, rk = [], []
    for i in range(y_true.shape[0]):
        truth = set(np.where(y_true[i]==1)[0])
        if not truth: continue
        topk = set(np.argsort(y_pred[i])[-k:][::-1])
        pk.append(len(topk & truth)/k)
        rk.append(len(topk & truth)/len(truth))
    return np.mean(pk) if pk else 0, np.mean(rk) if rk else 0

print("\n=== P@k / R@k (TH=0.3) ===", flush=True)
y_bin_opt = (val_preds > 0.3).astype(int)
micro_f1 = f1_score(val_targets, y_bin_opt, average='micro', zero_division=0)
macro_f1 = f1_score(val_targets, y_bin_opt, average='macro', zero_division=0)
print(f"  Micro F1={micro_f1:.4f}  Macro F1={macro_f1:.4f}")
for k in [5,8,10,15]:
    p,r = pk_rk(val_targets, val_preds, k)
    print(f"  P@{k}={p:.4f}  R@{k}={r:.4f}")

np.save('v3_longformer_y_true.npy', val_targets)
np.save('v3_longformer_y_pred.npy', val_preds)
print("\nDONE: v3_longformer_y_*.npy saved", flush=True)
