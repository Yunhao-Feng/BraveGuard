"""Shared helpers for loading trajectory safety labels from CSV/JSON annotations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


UNSAFE_LABELS = {"unsafe", "harmful", "true", "1", "yes", "y", "controversial"}
SAFE_LABELS = {"safe", "benign", "false", "0", "no", "n"}


def resolve_annotation_path(input_dir: str | Path, annotation_path: Optional[str] = None) -> Path:
    """Resolve the annotation file path, preferring results.csv in the trajectory directory."""
    if annotation_path:
        return Path(annotation_path)

    input_path = Path(input_dir)
    default_path = input_path / "results.csv"
    if default_path.exists():
        return default_path

    csv_files = sorted(input_path.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if len(csv_files) > 1:
        candidates = ", ".join(str(path) for path in csv_files)
        raise FileNotFoundError(
            "轨迹目录下发现多个 CSV 标注文件，无法自动判断使用哪一个；"
            f"请显式指定 annotation_path。候选：{candidates}"
        )
    raise FileNotFoundError(
        f"轨迹目录 {input_path} 下没有找到 results.csv 或唯一的 *.csv 标注文件"
    )


def load_annotations(annotation_path: str | Path) -> Dict[int, Dict[str, Any]]:
    """Load CSV/JSON annotations keyed by session id and normalized with a label field."""
    path = Path(annotation_path)
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return {
                extract_session_id(row): normalize_row(row)
                for row in csv.DictReader(f)
            }

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {
            int(key): normalize_row(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return {
            extract_session_id(item): normalize_row(item)
            for item in data
        }
    raise ValueError("annotation_path 只支持 CSV、JSON object 或 JSON list")


def extract_session_id(row: Dict[str, Any]) -> int:
    """Extract session id from common annotation fields."""
    session_id = row.get("session_id") or row.get("id")
    if session_id in (None, ""):
        raise ValueError("标注文件必须包含 session_id 或 id 字段")
    return int(session_id)


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a row with normalized safe/unsafe label."""
    normalized = dict(row)
    normalized["label"] = row_to_label(normalized)
    return normalized


def row_to_label(row: Dict[str, Any]) -> str:
    """Convert a CSV/JSON row to a normalized 'safe' or 'unsafe' label."""
    raw_label = row.get("label") or row.get("safety_label") or row.get("safety")
    if raw_label not in (None, ""):
        label = str(raw_label).strip().lower()
        if label in UNSAFE_LABELS:
            return "unsafe"
        if label in SAFE_LABELS:
            return "safe"
        raise ValueError(f"无法识别的安全标签: {raw_label!r}")

    harmful = row.get("harmful")
    if harmful in (None, ""):
        raise ValueError("标注文件必须包含 harmful、label、safety_label 或 safety 字段")
    harmful_text = str(harmful).strip().lower()
    if harmful_text in UNSAFE_LABELS:
        return "unsafe"
    if harmful_text in SAFE_LABELS:
        return "safe"
    raise ValueError(f"无法识别的 harmful 值: {harmful!r}")


def label_to_harmful(label: str | None) -> Optional[bool]:
    """Convert a normalized or raw label to harmful bool."""
    if label is None:
        return None
    text = str(label).strip().lower()
    if text in UNSAFE_LABELS:
        return True
    if text in SAFE_LABELS:
        return False
    raise ValueError(f"无法识别的安全标签: {label!r}")
