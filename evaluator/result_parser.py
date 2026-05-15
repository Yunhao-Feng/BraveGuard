"""结果解析模块：解析不同 Guard 模型的输出"""

import re
import logging
from typing import Optional

from .base_parser import BaseResultParser, GuardResult

logger = logging.getLogger(__name__)


_BINARY_LABEL_PATTERN = re.compile(
    r"(?:Safety\s*:\s*)?\b(unsafe|safe)\b",
    re.IGNORECASE,
)


def _parse_binary_label(raw_output: str) -> Optional[str]:
    """Extract a binary safe/unsafe label, accepting any letter casing."""
    match = _BINARY_LABEL_PATTERN.search(raw_output.strip())
    if not match:
        return None
    return match.group(1).lower()


class Qwen3ResultParser(BaseResultParser):
    """
    解析 Qwen3Guard 二分类输出。

    期望输出只包含 safe 或 unsafe；大小写不敏感，也兼容旧式
    ``Safety: Safe`` / ``Safety: Unsafe`` 前缀。Categories/Refusal 会被忽略。
    """

    def parse(self, session_id: int, raw_output: str) -> GuardResult:
        """解析单条模型输出"""
        safety_label = _parse_binary_label(raw_output)
        error = ""

        if safety_label is None:
            error = "无法解析二分类安全标签（期望 safe 或 unsafe）"
            logger.warning(f"Session {session_id}: {error}. 原始输出: {raw_output[:200]}")

        harmful = safety_label == "unsafe"

        return GuardResult(
            session_id=session_id,
            safety_label=safety_label,
            categories=None,
            refusal=None,
            raw_output=raw_output,
            harmful=harmful,
            error=error,
        )


class LlamaResultParser(BaseResultParser):
    """
    解析 LlamaGuard 二分类输出。

    期望输出只包含 safe 或 unsafe；大小写不敏感，也兼容额外文本中的第一个
    safe/unsafe token。Categories/Refusal 会被忽略。
    """

    def parse(self, session_id: int, raw_output: str) -> GuardResult:
        """解析 LlamaGuard 输出"""
        safety_label = _parse_binary_label(raw_output)
        error = ""

        if safety_label is None:
            error = "无法解析二分类安全标签（期望 safe 或 unsafe）"
            logger.warning(f"Session {session_id}: {error}. 原始输出: {raw_output[:200]}")

        harmful = safety_label == "unsafe"

        return GuardResult(
            session_id=session_id,
            safety_label=safety_label,
            categories=None,
            refusal=None,
            raw_output=raw_output,
            harmful=harmful,
            error=error,
        )
