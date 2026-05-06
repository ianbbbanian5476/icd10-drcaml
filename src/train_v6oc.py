#!/usr/bin/env python3
import os, json, torch, torch.nn as nn, numpy as np, sys
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 3e-5
LAMBDA_REG = 0.1
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

TRAIN_DATA_PATH = "../train_v5.json"
VAL_DATA_PATH = "../val_v5.json"
LABEL_MAP_PATH = "../label_map_v5.json"
EMBEDDINGS_PATH = "../v5_label_embeddings.pt"
BEST_MODEL_PATH = "v6oc_best_model.pth"

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

class BertDRCAML(nn.Module):
    def __init__(self, model_name, num_labels, label_embeddings):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.attention_weights = nn.Linear(hidden_size, num_labels)
        self.out_weights = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.out_bias = nn.Parameter(torch.Tensor(num_labels))
        nn.init.xavier_uniform_(self.out_weights)
        nn.init.zeros_(self.out_bias)
        self.register_buffer('target_embeddings', label_embeddings)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state
        A_logits = self.attention_weights(H)
        mask = attention_mask.unsqueeze(-1)
        A_logits = A_logits.masked_fill(mask == 0, -1e4)
        A = torch.softmax(A_logits, dim=1)
        V = torch.bmm(A.transpose(1, 2), H)
        logits = torch.sum(V * self.out_weights.unsqueeze(0), dim=-1) + self.out_bias
        return logits, self.out_weights

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

def calculate_pk_rk(y_true, y_pred, k):
    pk_list, rk_list = [], []
    for i in range(y_true.shape[0]):
        true_labels = set(np.where(y_true[i] == 1)[0])
        if len(true_labels) == 0:
            continue
        top_k_preds = set(np.argsort(y_pred[i])[-k:][::-1])
        hits = len(top_k_preds.intersection(true_labels))
        pk_list.append(hits / k)
        rk_list.append(hits / len(true_labels))
    return (np.mean(pk_list) if pk_list else 0.0), (np.mean(rk_list) if rk_list else 0.0)

def quick_lr_test(model, train_loader, val_loader, lrs):
    best_lr, best_f1 = lrs[0], 0
    for lr in lrs:
        test_model = BertDRCAML(MODEL_NAME, NUM_LABELS, label_embeddings).to(device)
        test_model.load_state_dict(model.state_dict())
        opt = AdamW(test_model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        test_model.train()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 50:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            opt.zero_grad()
            logits, _ = test_model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
        test_model.eval()
        targets, preds = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 20:
                    break
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                logits, _ = test_model(input_ids, attention_mask)
                targets.extend(labels.cpu().numpy())
                preds.extend(torch.sigmoid(logits).cpu().numpy())
        targets = np.array(targets)
        preds_bin = (np.array(preds) > 0.5).astype(int)
        f1 = f1_score(targets, preds_bin, average='micro', zero_division=0)
        print(f"  LR={lr:.0e} → Micro F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_lr = lr
    return best_lr

def main():
    global NUM_LABELS, label_embeddings

    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    label2id = raw.get("label2id", raw)
    NUM_LABELS = len(label2id)
    print(f"Labels: {NUM_LABELS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_embeddings = torch.load(EMBEDDINGS_PATH, map_location=device)

    train_dataset = ICD10Dataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN, label2id)
    val_dataset = ICD10Dataset(VAL_DATA_PATH, tokenizer, MAX_LEN, label2id)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = BertDRCAML(MODEL_NAME, NUM_LABELS, label_embeddings).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    focal_criterion = FocalBCELoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    mse_criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    best_val_f1 = 0.0

    print(f"\nV6OC Training | λ={LAMBDA_REG} | Focal(α={FOCAL_ALPHA},γ={FOCAL_GAMMA}) | LR={LEARNING_RATE:.0e}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_focal, total_reg = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, current_weights = model(input_ids, attention_mask)
                loss_focal = focal_criterion(logits, labels)
                loss_reg = mse_criterion(current_weights, model.target_embeddings)
                loss = loss_focal + LAMBDA_REG * loss_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_focal += loss_focal.item()
            total_reg += loss_reg.item()
            loop.set_postfix(Focal=f"{loss_focal.item():.4f}", REG=f"{(LAMBDA_REG*loss_reg).item():.4f}")

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

        micro_f1 = f1_score(val_targets, val_preds_bin, average='micro', zero_division=0)
        macro_f1 = f1_score(val_targets, val_preds_bin, average='macro', zero_division=0)
        micro_auc = roc_auc_score(val_targets, val_preds, average='micro')
        print(f"  F1: micro={micro_f1:.4f} macro={macro_f1:.4f} AUC={micro_auc:.4f}")

        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✅ Best model saved (Micro F1={best_val_f1:.4f})")

    print(f"\nLoading best model (F1={best_val_f1:.4f})...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    val_targets, val_preds = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            with torch.cuda.amp.autocast():
                logits, _ = model(input_ids, attention_mask)
            val_targets.extend(labels.cpu().numpy())
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())

    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)
    val_preds_bin = (val_preds > 0.5).astype(int)

    micro_f1 = f1_score(val_targets, val_preds_bin, average='micro', zero_division=0)
    macro_f1 = f1_score(val_targets, val_preds_bin, average='macro', zero_division=0)
    micro_auc = roc_auc_score(val_targets, val_preds, average='micro')
    p5, r5 = calculate_pk_rk(val_targets, val_preds, k=5)
    p8, r8 = calculate_pk_rk(val_targets, val_preds, k=8)
    p10, r10 = calculate_pk_rk(val_targets, val_preds, k=10)
    p15, r15 = calculate_pk_rk(val_targets, val_preds, k=15)

    np.save('v6oc_y_true.npy', val_targets)
    np.save('v6oc_y_pred.npy', val_preds)

    print(f"\n{'='*50}")
    print(f"V6OC Final Results")
    print(f"{'='*50}")
    print(f"  Micro F1:  {micro_f1:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Micro AUC: {micro_auc:.4f}")
    print(f"  P@5:  {p5:.4f}  |  R@5:  {r5:.4f}")
    print(f"  P@8:  {p8:.4f}  |  R@8:  {r8:.4f}")
    print(f"  P@10: {p10:.4f}  |  R@10: {r10:.4f}")
    print(f"  P@15: {p15:.4f}  |  R@15: {r15:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
