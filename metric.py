import csv

filename = "guard_sft/merged_mode3.csv"

total = 0
correct = 0

with open(filename, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        harmful = row["harmful"].strip().lower() == "true"
        expected = row["expected_harmful"].strip().lower() == "true"
        total += 1
        if harmful == expected:
            correct += 1

acc = correct / total if total > 0 else 0
print(f"ACC = {acc:.4f}")
