"""Reference-label loading and classification metrics for guard evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


LabelMap = dict[int, bool]


def resolve_annotation_path(input_dir: str, annotation_path: str | None = None) -> Path | None:
    """Resolve an optional reference-label file for evaluation metrics."""
    if annotation_path:
        path = Path(annotation_path)
        return path if path.exists() else None

    default_path = Path(input_dir) / "results.csv"
    if default_path.exists():
        return default_path
    return None


def load_reference_labels(input_dir: str, annotation_path: str | None = None) -> LabelMap:
    """Load session_id -> harmful labels from CSV/JSON annotations."""
    path = resolve_annotation_path(input_dir, annotation_path)
    if path is None:
        return {}

    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return {
                _extract_session_id(row): _row_to_harmful(row)
                for row in csv.DictReader(f)
            }

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {int(key): _row_to_harmful(value) for key, value in data.items()}
    if isinstance(data, list):
        return {_extract_session_id(item): _row_to_harmful(item) for item in data}
    raise ValueError("annotation_path 只支持 CSV、JSON object 或 JSON list")


def _extract_session_id(row: dict[str, Any]) -> int:
    session_id = row.get("session_id") or row.get("id")
    if session_id in (None, ""):
        raise ValueError("标注文件必须包含 session_id 或 id 字段")
    return int(session_id)


def _row_to_harmful(row: dict[str, Any]) -> bool:
    raw_label = row.get("label") or row.get("safety_label") or row.get("safety")
    if raw_label not in (None, ""):
        label = str(raw_label).strip().lower()
        if label in {"unsafe", "harmful", "true", "1", "yes"}:
            return True
        if label in {"safe", "benign", "false", "0", "no"}:
            return False
        raise ValueError(f"无法识别的安全标签: {raw_label!r}")

    harmful = row.get("harmful")
    if harmful in (None, ""):
        raise ValueError("标注文件必须包含 harmful、label、safety_label 或 safety 字段")
    harmful_text = str(harmful).strip().lower()
    if harmful_text in {"true", "1", "yes", "y", "unsafe", "harmful"}:
        return True
    if harmful_text in {"false", "0", "no", "n", "safe", "benign"}:
        return False
    raise ValueError(f"无法识别的 harmful 值: {harmful!r}")


class BinaryMetrics:
    """Accumulate binary harmful/safe classification metrics."""

    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.count = 0

    def update(self, expected_harmful: bool, predicted_harmful: bool) -> None:
        self.count += 1
        if expected_harmful and predicted_harmful:
            self.tp += 1
        elif not expected_harmful and not predicted_harmful:
            self.tn += 1
        elif not expected_harmful and predicted_harmful:
            self.fp += 1
        else:
            self.fn += 1

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.count if self.count else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0
