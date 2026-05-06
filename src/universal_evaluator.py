#!/usr/bin/env python3
import os, sys, json, torch, numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import argparse

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        A_logits = A_logits.masked_fill(mask == 0, -1e4)
        A = torch.softmax(A_logits, dim=1)
        V = torch.bmm(A.transpose(1, 2), H)
        logits = torch.sum(V * self.out_weights.unsqueeze(0), dim=-1) + self.out_bias
        return logits, self.out_weights

class BertCLS(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits, None

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
        raw_labels = item.get('all_codes', item.get('label', item.get('codes', [])))

        labels = torch.zeros(self.num_labels)
        for code in raw_labels:
            if isinstance(code, str) and code in self.label2id:
                labels[self.label2id[code]] = 1.0
            elif isinstance(code, (int, float)) and 0 <= code < self.num_labels:
                labels[int(code)] = 1.0
            elif isinstance(code, str) and code.isdigit():
                idx_int = int(code)
                if 0 <= idx_int < self.num_labels:
                    labels[idx_int] = 1.0

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['cls', 'drcaml'])
    parser.add_argument('--model_name', default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--model_weights', required=True)
    parser.add_argument('--label_map', required=True)
    parser.add_argument('--val_data', required=True)
    parser.add_argument('--embeddings', default=None)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_prefix', default='eval_result')
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"🔬 通用評估器啟動")
    print(f"   Model type: {args.model_type}")
    print(f"   Model weights: {args.model_weights}")
    print(f"   Label map: {args.label_map}")
    print(f"   Val data: {args.val_data}")
    print(f"   MAX_LEN: {args.max_len}")
    print(f"{'='*70}")

    with open(args.label_map, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    label2id = raw.get("label2id", raw)
    NUM_LABELS = len(label2id)
    print(f"📊 標籤數量: {NUM_LABELS}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.model_type == 'drcaml':
        if not args.embeddings:
            print("❌ DR-CAML 需要 --embeddings")
            sys.exit(1)
        print(f"📥 載入 embedding: {args.embeddings}")
        label_embeddings = torch.load(args.embeddings, map_location=device)
        model = BertDRCAML(args.model_name, NUM_LABELS, label_embeddings)
    else:
        model = BertCLS(args.model_name, NUM_LABELS)

    print(f"📥 載入權重: {args.model_weights}")
    state_dict = torch.load(args.model_weights, map_location=device)
    new_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    if args.model_type == 'cls':
        model.model.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(new_state_dict, strict=True)

    model.to(device)
    model.eval()

    val_dataset = ICD10Dataset(args.val_data, tokenizer, args.max_len, label2id)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    val_targets, val_preds = [], []
    print(f"🔍 推論 {len(val_dataset)} 筆...")
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

    micro_f1 = f1_score(val_targets, val_preds_bin, average='micro', zero_division=0)
    macro_f1 = f1_score(val_targets, val_preds_bin, average='macro', zero_division=0)
    micro_auc = roc_auc_score(val_targets, val_preds, average='micro')

    p5, r5 = calculate_pk_rk(val_targets, val_preds, k=5)
    p8, r8 = calculate_pk_rk(val_targets, val_preds, k=8)
    p10, r10 = calculate_pk_rk(val_targets, val_preds, k=10)
    p15, r15 = calculate_pk_rk(val_targets, val_preds, k=15)

    np.save(f'{args.output_prefix}_y_true.npy', val_targets)
    np.save(f'{args.output_prefix}_y_pred.npy', val_preds)

    print(f"\n{'='*50}")
    print(f"📊 評估結果")
    print(f"{'='*50}")
    print(f"  Micro F1:  {micro_f1:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Micro AUC: {micro_auc:.4f}")
    print(f"  P@5:  {p5:.4f}  |  R@5:  {r5:.4f}")
    print(f"  P@8:  {p8:.4f}  |  R@8:  {r8:.4f}")
    print(f"  P@10: {p10:.4f}  |  R@10: {r10:.4f}")
    print(f"  P@15: {p15:.4f}  |  R@15: {r15:.4f}")
    print(f"{'='*50}")

    csv_line = f"{args.model_type},{args.model_weights},{args.val_data},{NUM_LABELS},{micro_f1:.4f},{macro_f1:.4f},{micro_auc:.4f},{p5:.4f},{r5:.4f},{p8:.4f},{r8:.4f},{p10:.4f},{r10:.4f},{p15:.4f},{r15:.4f}"
    with open(f'{args.output_prefix}_result.csv', 'w') as f:
        f.write("model_type,model_weights,val_data,num_labels,micro_f1,macro_f1,micro_auc,p_at_5,r_at_5,p_at_8,r_at_8,p_at_10,r_at_10,p_at_15,r_at_15\n")
        f.write(csv_line + "\n")

    print(f"✅ 結果已輸出: {args.output_prefix}_result.csv")

if __name__ == "__main__":
    main()
