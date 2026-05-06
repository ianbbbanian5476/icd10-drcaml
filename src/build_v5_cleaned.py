import json
from collections import Counter

# ================= 參數設定 =================
LABEL_MAP_FILE = 'label_map_v4.json'
TRAIN_FILE = 'train_v4.json'
VAL_FILE = 'val_v4.json'

MIN_FREQ = 50

# 🌟 終極 V5 版輸出檔
OUTPUT_LABEL_MAP = 'label_map_v5.json'
OUTPUT_TRAIN = 'train_v5.json'
OUTPUT_VAL = 'val_v5.json'
# ===========================================

def main():
    print("🚀 [V5 終極版資料煉成陣] 啟動！(4碼降維 + 頻率門檻 50)")
    
    # 1. 載入 V4 字典與資料
    with open(LABEL_MAP_FILE, 'r', encoding='utf-8') as f:
        id2label_v4 = json.load(f)['id2label']
        
    print(f"📂 正在讀取 V4 資料...")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
        
    all_data = train_data + val_data
    print(f"   => 成功載入 {len(all_data):,} 筆病歷！")

    # 2. 執行「降維至 4 碼」並統計頻率
    print("\n🗜️ 1. 正在執行 ICD-10 降維至 4 碼...")
    rollup_4_counter = Counter()
    for record in all_data:
        codes = [id2label_v4[str(lbl_id)] for lbl_id in record['label']]
        # 降維邏輯：去除小數點，取前 4 個字元，並使用 set 確保同一個病人在同一個 4 碼分類下只算一次
        codes_4d = list(set([c.replace('.', '').strip()[:4] for c in codes]))
        rollup_4_counter.update(codes_4d)

    # 3. 執行「門檻過濾 (MIN_FREQ = 50)」
    print(f"✂️ 2. 正在剔除頻率低於 {MIN_FREQ} 次的罕見 4 碼疾病...")
    valid_codes = [code for code, count in rollup_4_counter.items() if count >= MIN_FREQ]
    valid_codes.sort()
    
    # 4. 建立 V5 終極字典
    label2id_v5 = {code: idx for idx, code in enumerate(valid_codes)}
    id2label_v5 = {str(idx): code for idx, code in enumerate(valid_codes)}
    
    with open(OUTPUT_LABEL_MAP, 'w', encoding='utf-8') as f:
        json.dump({"label2id": label2id_v5, "id2label": id2label_v5}, f, ensure_ascii=False, indent=4)
        
    print(f"   => 📖 V5 字典建立完成！總計 【 {len(valid_codes):,} 】 種黃金代碼。")

    # 5. 轉換病歷格式 (並丟棄變成 0 標籤的病歷)
    print("\n⚙️ 3. 正在將病歷轉換為 V5 格式...")
    def process_data(data):
        processed = []
        dropped = 0
        for record in data:
            codes = [id2label_v4[str(lbl_id)] for lbl_id in record['label']]
            codes_4d = list(set([c.replace('.', '').strip()[:4] for c in codes]))
            
            # 只保留存在於 V5 黃金字典中的代碼
            valid_ids = [label2id_v5[c] for c in codes_4d if c in label2id_v5]
            
            if len(valid_ids) > 0:
                processed.append({"text": record['text'], "label": valid_ids})
            else:
                dropped += 1
        return processed, dropped

    train_v5, train_dropped = process_data(train_data)
    val_v5, val_dropped = process_data(val_data)
    total_dropped = train_dropped + val_dropped

    # 6. 儲存檔案
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_v5, f, ensure_ascii=False)
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        json.dump(val_v5, f, ensure_ascii=False)

    print("\n================ 🏆 V5 終極版資料準備完成 ================")
    print(f"   📉 疾病種類數 : 14,219 類 ➡️  {len(valid_codes):,} 類")
    print(f"   🚨 因全數罕見而丟棄的病歷: {total_dropped:,} 筆 ({total_dropped/len(all_data)*100:.2f}%)")
    print("-" * 50)
    print(f"   📦 訓練集 (Train): {len(train_v5):,} 筆 -> {OUTPUT_TRAIN}")
    print(f"   📝 驗證集 (Val)  : {len(val_v5):,} 筆 -> {OUTPUT_VAL}")
    print(f"   📖 新字典        : {len(valid_codes):,} 類 -> {OUTPUT_LABEL_MAP}")
    print("==========================================================")

if __name__ == "__main__":
    main()