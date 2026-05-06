import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================= 參數設定 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 目前使用的運算設備: {device}")

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
LABEL_MAP_PATH = "label_map_v6_3.json"               # 👈 改成未清洗的字典
ICD10_DICT_PATH = "icd10_clean_final.json"
OUTPUT_PT_PATH = "mimic_v6_3_embeddings.pt"          # 👈 新的 Tensor 存檔名稱
OUTPUT_JSON_PATH = "v6_3_label_descriptions.json"
# ============================================

def load_icd10_dictionary(dict_path):
    """載入官方 ICD-10 字典，並移除小數點建立純粹的 alphanumeric 對應表"""
    print(f"📂 正在載入官方 ICD-10 字典 ({dict_path})...")
    with open(dict_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    icd_dict = {}
    for item in data:
        code = str(item.get('code', '')).strip().upper()
        title = item.get('title', '').strip()
        if code and title:
            # 【關鍵修復】：將官方代碼 A02.0 轉換為 A020
            clean_code = code.replace('.', '')
            icd_dict[clean_code] = title
            
            # 為了防呆，保留一份原本有小數點的版本也存進去
            icd_dict[code] = title
            
    print(f"✅ 成功建立正規化字典，共包含 {len(icd_dict):,} 筆定義。")
    return icd_dict

def main():
    print(f"\n{'='*70}")
    print(f"🚀 V6 DR-CAML 階段一：建立疾病標準答案庫 (Label Embeddings) [修正版]")
    print(f"{'='*70}")

    # 1. 讀取 V5 標籤對應表
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        raw_label_map = json.load(f)
        
    if "label2id" in raw_label_map:
        label2id = raw_label_map["label2id"]
    else:
        label2id = raw_label_map
        
    num_labels = len(label2id)
    print(f"📌 載入 V5 標籤共 {num_labels} 類。")

    # 2. 載入官方字典 (已修復小數點問題)
    icd_dict = load_icd10_dictionary(ICD10_DICT_PATH)

    # 3. 為 2099 個標籤尋找官方文字定義
    # 確保矩陣順序與 label2id 完全一致 (0 到 2098)
    ordered_codes = [None] * num_labels
    for code, idx in label2id.items():
        # 確保 V5 標籤也被清除了小數點 (以防萬一)
        ordered_codes[idx] = code.upper().replace('.', '')

    descriptions = []
    missing_count = 0
    
    print("\n🔍 開始為 V5 標籤配對官方文字定義...")
    for code in ordered_codes:
        # 【關鍵修復】：經過正規化後，現在可以 100% 精確命中！
        if code in icd_dict:
            descriptions.append(icd_dict[code])
        else:
            # 如果真的連拔掉小數點都找不到，再使用容錯比對
            candidates = [k for k in icd_dict.keys() if code.startswith(k) or k.startswith(code)]
            if candidates:
                best_match = min(candidates, key=lambda k: abs(len(k) - len(code)))
                descriptions.append(icd_dict[best_match])
            else:
                descriptions.append(f"Disease code {code}")
                missing_count += 1
                
    print(f"✅ 配對完成！(完美命中或近似命中: {num_labels - missing_count} 筆，未命中: {missing_count} 筆)")
    
    # 儲存一份文字版的對照表供未來人類檢視用
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump({"ordered_codes": ordered_codes, "descriptions": descriptions}, f, indent=4)

    # 4. 啟動 Bio_ClinicalBERT 進行語意轉換
    print(f"\n🧠 啟動 {MODEL_NAME} 將定義轉換為高維向量...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 準備一個空的 Tensor 來裝所有向量 [2099, 768]
    hidden_size = model.config.hidden_size
    label_embeddings = torch.zeros((num_labels, hidden_size))

    # 使用 Batch 的方式來加速推論
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, num_labels, batch_size), desc="生成 Embeddings"):
            batch_texts = descriptions[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=64, 
                return_tensors='pt'
            ).to(device)

            # 通過 BERT
            outputs = model(**encoded)
            
            # 提取 [CLS] token 的向量作為整句定義的「語意標準答案」
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # 將算好的向量存回大矩陣
            label_embeddings[i:i+batch_size] = cls_embeddings.cpu()

    # 5. 存檔
    torch.save(label_embeddings, OUTPUT_PT_PATH)
    print(f"\n🎉 階段一完成！")
    print(f"📦 產出檔案 1: {OUTPUT_PT_PATH} (這就是 DR-CAML 要用的神經網路標準答案，維度: {label_embeddings.shape})")
    print(f"📦 產出檔案 2: {OUTPUT_JSON_PATH} (供您檢查模型到底讀了什麼文字)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()