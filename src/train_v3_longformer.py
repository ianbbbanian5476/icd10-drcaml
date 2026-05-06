import os
import sys
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint

# ================= vGPU 相容性設定 =================
print("[系統] 套用 vGPU 相容性設定...")
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# ===================================================

# ================= 參數設定 (V3.1 Longformer 智慧閱讀版) =================
TRAIN_FILE = 'train_v2.json'           
VAL_FILE = 'val_v2.json'               
LABEL_MAP_FILE = 'label_map_v2.json'   
OUTPUT_DIR = './clinical_longformer_v3_model' 
MODEL_NAME = "yikuan8/Clinical-Longformer"

# 🔥 秘技 1：長度砍半，運算時間大幅縮短！
MAX_LEN = 1024 

# 🔥 秘技 2：記憶體還有剩，加大 Batch 塞滿它！
BATCH_SIZE = 48            
ACCUMULATION_STEPS = 2   # 8 * 4 = 32
EPOCHS = 3           
EXAM_SIZE = 10000         
# ======================================================

class MultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, num_labels):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_labels = num_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item.get('text', ''))
        labels_list = item.get('label', item.get('labels', []))
        
        if not isinstance(labels_list, list):
            labels_list = [labels_list]

        label_tensor = torch.zeros(self.num_labels, dtype=torch.float)
        for l in labels_list:
            try: label_tensor[int(l)] = 1.0
            except: pass 

        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length',
            max_length=self.max_len, 
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor 
        }

def main():
    print(f"1. 正在載入 V2 字典 {LABEL_MAP_FILE}...")
    with open(LABEL_MAP_FILE, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    num_labels = len(label_map['label2id'])
    
    print(f"2. 正在初始化 Longformer 模型 {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 🔥 秘技 3：黃金設定！設定左側截斷，捨棄冗長主訴，直搗最後的診斷！
    tokenizer.truncation_side = 'left'

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        problem_type="multi_label_classification", 
        ignore_mismatched_sizes=True
    )

    print(f"3. 正在將病歷轉換為模型格式 (MAX_LEN={MAX_LEN}, 截斷模式: 左側截斷)...")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    if len(val_data) > EXAM_SIZE:
        random.seed(42)
        val_data = random.sample(val_data, EXAM_SIZE)

    train_dataset = MultiLabelDataset(train_data, tokenizer, MAX_LEN, num_labels)
    val_dataset = MultiLabelDataset(val_data, tokenizer, MAX_LEN, num_labels)

    print("4. 設定訓練管線 (Trainer)...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_v3',
        logging_steps=50, 
        
        evaluation_strategy="epoch",  
        save_strategy="epoch",        
        save_total_limit=2,  
        
        fp16=True, 
        # 🔥 解除 CPU 瓶頸，讓 4 個小幫手全力餵資料給 GPU
        dataloader_num_workers=4, 
        gradient_checkpointing=True,
        
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("🚀 5. 準備開始 V3.1 Longformer 智慧閱讀版訓練！")
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
        
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ V3.1 訓練完畢！")

if __name__ == "__main__":
    main()