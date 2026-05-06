#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

MIN_FREQ = 20
BASE = Path("/home/s11350310/icd10-2nd/BERT_TRY/V2")
OUT = Path("/home/s11350310/icd10-2nd/BERT_TRY/V2/opencode_fix")

def clean_code(code):
    return code.replace('.', '')[:4]

def build_split(data_path, code_counts, id2label, min_freq):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = []
    skipped = 0
    for item in data:
        old_labels = item.get('label', [])
        new_labels = set()
        for label_idx in old_labels:
            code_str = id2label.get(str(label_idx), id2label.get(label_idx))
            if code_str is None:
                continue
            cc = clean_code(str(code_str))
            if code_counts.get(cc, 0) >= min_freq:
                new_labels.add(cc)

        if not new_labels:
            skipped += 1
            continue
        cleaned.append({'text': item['text'], 'label': sorted(new_labels)})

    print(f"  Input: {len(data)} → Output: {len(cleaned)} (skipped {skipped} with no valid labels)")
    return cleaned

print("Step 1: Load V4 label map and count code frequencies...")
with open(BASE / "label_map_v4.json", 'r', encoding='utf-8') as f:
    v4_map = json.load(f)
label2id = v4_map.get("label2id", v4_map)

code_counts = Counter()
id2label = v4_map.get("id2label", {})

with open(BASE / "train_v4.json", 'r', encoding='utf-8') as f:
    train_v4 = json.load(f)

for item in train_v4:
    for label_idx in item.get('label', []):
        code_str = id2label.get(str(label_idx), id2label.get(label_idx))
        if code_str is None:
            continue
        cc = clean_code(str(code_str))
        code_counts[cc] += 1

valid_codes = {c for c, cnt in code_counts.items() if cnt >= MIN_FREQ}
print(f"  V4 codes: {len(code_counts)} → After MIN_FREQ={MIN_FREQ}: {len(valid_codes)}")

print(f"\nStep 2: Build label map (MIN_FREQ={MIN_FREQ})...")
new_label_map = {"label2id": {}, "id2label": {}}
for idx, code in enumerate(sorted(valid_codes)):
    new_label_map["label2id"][code] = idx
    new_label_map["id2label"][str(idx)] = code

with open(OUT / f"label_map_mf20.json", 'w', encoding='utf-8') as f:
    json.dump(new_label_map, f, ensure_ascii=False, indent=2)
print(f"  Labels: {len(valid_codes)}")

print(f"\nStep 3: Rebuild train set...")
train_data = build_split(BASE / "train_v4.json", dict.fromkeys(valid_codes, MIN_FREQ), id2label, MIN_FREQ)
with open(OUT / "train_mf20.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, separators=(',', ':'))

print(f"\nStep 4: Rebuild val set...")
val_data = build_split(BASE / "val_v4.json", dict.fromkeys(valid_codes, MIN_FREQ), id2label, MIN_FREQ)
with open(OUT / "val_mf20.json", 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, separators=(',', ':'))

print(f"\nMIN_FREQ={MIN_FREQ} data ready: {len(train_data)} train, {len(val_data)} val, {len(valid_codes)} labels")
