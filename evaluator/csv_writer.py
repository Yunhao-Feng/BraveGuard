"""CSV 输出模块"""

import csv
import logging
import threading
from pathlib import Path
from typing import List

from .result_parser import GuardResult

logger = logging.getLogger(__name__)

# CSV 字段定义
CSV_FIELDS = [
    "session_id",
    "harmful",
    "safety_label",
    "categories",
    "refusal",
    "raw_output",
    "error",
]


class CSVWriter:
    """写入评估结果到 CSV 文件"""

    def __init__(self, output_path: str | None):
        self.output_path = Path(output_path) if output_path else None
        self._lock = threading.Lock()  # 保证追加写入的线程安全
        self._initialized = False
        self._enabled = output_path is not None

    def write(self, results: List[GuardResult]):
        """写入所有结果"""
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 按 session_id 排序
        sorted_results = sorted(results, key=lambda r: r.session_id)

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for result in sorted_results:
                row = result.to_dict()
                writer.writerow(row)

        logger.info(f"结果已写入: {self.output_path}")

        # 打印统计
        total = len(sorted_results)
        harmful_count = sum(1 for r in sorted_results if r.harmful)
        error_count = sum(1 for r in sorted_results if r.error)
        if total > 0:
            asr = harmful_count / total
            print(f"\n[统计] 总数: {total}, 有害: {harmful_count}, "
                  f"ASR: {asr:.4f}, 解析错误: {error_count}")
        else:
            print("\n[统计] 无结果")

    def initialize(self):
        """初始化 CSV 文件（写入 header）"""
        if not self._enabled:
            return

        with self._lock:
            if self._initialized:
                return

            # 确保输出目录存在
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入 header
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()

            self._initialized = True
            logger.info(f"CSV 文件已初始化: {self.output_path}")

    def append(self, result: GuardResult):
        """追加单条结果到 CSV（线程安全）"""
        if not self._enabled:
            return

        with self._lock:
            # 确保文件已初始化
            if not self._initialized:
                self.initialize()

            # 追加写入
            with open(self.output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(result.to_dict())
