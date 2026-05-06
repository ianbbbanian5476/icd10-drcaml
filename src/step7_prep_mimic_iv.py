import pandas as pd
import json
from collections import Counter
import os

# ================= 檔案路徑設定 =================
NOTES_FILE = "discharge.csv"
DIAGNOSES_FILE = "diagnoses_icd.csv"

# ================= 清洗規則函數 =================
def clean_code(code):
    """東海 V6.4 清洗邏輯：移除小數點並取前四碼"""
    code = str(code).replace('.', '').strip()
    return code[:4] if len(code) >= 4 else code

def main():
    print("🚀 啟動 MIMIC-IV 資料前處理引擎...")
    
    # 檢查檔案是否存在
    if not os.path.exists(NOTES_FILE) or not os.path.exists(DIAGNOSES_FILE):
        print(f"❌ 找不到 CSV 檔案！請確保 {NOTES_FILE} 和 {DIAGNOSES_FILE} 在同一目錄下。")
        return

    # ---------------------------------------------------------
    # 1. 讀取與過濾診斷代碼 (diagnoses_icd.csv)
    # ---------------------------------------------------------
    print("\n📂 1. 載入診斷代碼 (Diagnoses)...")
    df_diag = pd.read_csv(DIAGNOSES_FILE, usecols=['hadm_id', 'icd_code', 'icd_version'])
    
    # MIMIC-IV 混雜了 ICD-9 和 ICD-10，我們只保留 ICD-10 以對齊榮總的實驗
    df_diag = df_diag[df_diag['icd_version'] == 10].dropna(subset=['icd_code'])
    print(f"   ➤ 過濾後共有 {len(df_diag):,} 筆 ICD-10 診斷紀錄。")

    # ---------------------------------------------------------
    # 2. 讀取病歷文字 (discharge.csv)
    # ---------------------------------------------------------
    print("📂 2. 載入出院摘要 (Notes) - 這可能需要幾分鐘，請稍候...")
    # 只讀取需要的欄位以節省 RAM
    df_notes = pd.read_csv(NOTES_FILE, usecols=['hadm_id', 'text']).dropna()
    
    # 如果同一個住院ID有多份摘要，將其合併 (雖然通常 discharge 只有一份)
    df_notes = df_notes.groupby('hadm_id')['text'].apply(lambda x: ' '.join(x)).reset_index()
    print(f"   ➤ 共有 {len(df_notes):,} 份獨立的出院摘要。")

    # =========================================================
    # ⚔️ 任務 A: 構建 V6.3 (未清洗版 / 原汁原味)
    # =========================================================
    print("\n⚔️ 3. 正在構建 V6.3 (未清洗版) 資料集...")
    # 將原始 ICD-10 代碼按 hadm_id 聚合
    v63_grouped = df_diag.groupby('hadm_id')['icd_code'].apply(list).reset_index()
    v63_merged = pd.merge(df_notes, v63_grouped, on='hadm_id', how='inner')
    
    v63_dataset = []
    v63_all_codes = set()
    for _, row in v63_merged.iterrows():
        codes = row['icd_code']
        v63_dataset.append({'text': row['text'], 'all_codes': codes})
        v63_all_codes.update(codes)
        
    v63_label2id = {code: i for i, code in enumerate(sorted(list(v63_all_codes)))}
    
    print(f"   👻 V6.3 產出：共 {len(v63_dataset):,} 筆病歷，標籤空間爆增至 {len(v63_label2id):,} 類！")
    with open('train_v6_3_mimic_full.json', 'w', encoding='utf-8') as f: json.dump(v63_dataset, f)
    with open('label_map_v6_3.json', 'w', encoding='utf-8') as f: json.dump(v63_label2id, f)

    # =========================================================
    # 🛡️ 任務 B: 構建 V6.4 (東海標準清洗版)
    # =========================================================
    print("\n🛡️ 4. 正在構建 V6.4 (東海清洗版) 資料集...")
    
    # B-1: 降維到前四碼
    df_diag['cleaned_code'] = df_diag['icd_code'].apply(clean_code)
    
    # B-2: 排除精神科 F 碼
    df_diag_clean = df_diag[~df_diag['cleaned_code'].str.startswith('F')]
    print(f"   ➤ 排除 F 碼後，剩餘 {len(df_diag_clean):,} 筆紀錄。")
    
    # B-3: 罕見病過濾 (出現次數 < 50 的標籤剔除)
    code_counts = Counter(df_diag_clean['cleaned_code'])
    valid_codes = {code for code, count in code_counts.items() if count >= 50}
    df_diag_clean = df_diag_clean[df_diag_clean['cleaned_code'].isin(valid_codes)]
    print(f"   ➤ 剔除 < 50 次的罕見病後，標籤種類收斂至 {len(valid_codes):,} 類。")
    
    # 聚合與合併
    v64_grouped = df_diag_clean.groupby('hadm_id')['cleaned_code'].apply(lambda x: list(set(x))).reset_index()
    v64_merged = pd.merge(df_notes, v64_grouped, on='hadm_id', how='inner')
    
    v64_dataset = []
    v64_all_codes = set()
    for _, row in v64_merged.iterrows():
        if len(row['cleaned_code']) > 0:
            v64_dataset.append({'text': row['text'], 'all_codes': row['cleaned_code']})
            v64_all_codes.update(row['cleaned_code'])

    v64_label2id = {code: i for i, code in enumerate(sorted(list(v64_all_codes)))}
    
    print(f"   ✨ V6.4 產出：共 {len(v64_dataset):,} 筆病歷，標籤空間完美收斂至 {len(v64_label2id):,} 類！")
    with open('train_v6_4_mimic_clean.json', 'w', encoding='utf-8') as f: json.dump(v64_dataset, f)
    with open('label_map_v6_4.json', 'w', encoding='utf-8') as f: json.dump(v64_label2id, f)

    print("\n✅ 所有前處理作業完成！彈藥已上膛，隨時可進行 V6 與純 CLS 訓練。")

if __name__ == "__main__":
    main()
