"""数据加载模块：负责读取 JSONL 轨迹文件和数据集元信息"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrajectoryLoader:
    """加载轨迹 JSONL 文件"""

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

    def list_session_files(self) -> List[Path]:
        """列出所有 session_item-*.jsonl 文件"""
        files = sorted(
            self.input_dir.glob("session_item-*.jsonl"),
            key=lambda p: int(p.stem.split("-")[1])
        )
        logger.info(f"找到 {len(files)} 个轨迹文件")
        return files

    def load_trajectory(self, filepath: Path) -> List[Dict]:
        """加载单个 JSONL 文件中的所有记录"""
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def extract_session_id(self, filepath: Path) -> int:
        """从文件名提取 session id"""
        # session_item-9.jsonl -> 9
        return int(filepath.stem.split("-")[1])


class DatasetLoader:
    """加载数据集元信息（target, comment, jailbreak_method, category）"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._cache: Optional[Dict[int, Dict]] = None

    def load(self) -> Dict[int, Dict]:
        """加载数据集并按 id 索引"""
        if self._cache is not None:
            return self._cache

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._cache = {item["id"]: item for item in data}
        logger.info(f"加载数据集: {len(self._cache)} 条记录")
        return self._cache

    def get_metadata(self, session_id: int) -> Optional[Dict]:
        """获取指定 session 的元信息"""
        dataset = self.load()
        return dataset.get(session_id)
