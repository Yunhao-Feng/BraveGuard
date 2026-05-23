#!/usr/bin/env python3
import json
import csv
import os
from pathlib import Path

# 1. 获取exports中所有存在的轨迹文件的id
exports_dir = Path("exports")
valid_ids = set()
for jsonl_file in exports_dir.glob("session_item-*.jsonl"):
    id_str = jsonl_file.stem.replace("session_item-", "")
    valid_ids.add(int(id_str))

print(f"找到 {len(valid_ids)} 个轨迹文件")

# 2. 清理 results.csv
csv_path = "exports/results.csv"
rows_to_keep = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    for row in reader:
        if int(row['id']) in valid_ids:
            rows_to_keep.append(row)

with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows_to_keep)

print(f"CSV: {len(rows_to_keep)} 条记录保留")

# 3. 清理 subset.json
json_path = "data/subset.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

filtered_data = [item for item in data if item['id'] in valid_ids]

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"JSON: {len(filtered_data)} 条记录保留")
print(f"\n✓ 完成！三者已对齐：{len(valid_ids)} 条")
