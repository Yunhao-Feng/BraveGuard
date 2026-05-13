"""结果解析模块：解析 Qwen3Guard 模型输出"""

import re
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """单条评估结果"""

    session_id: int
    safety_label: Optional[str]  # Safe / Unsafe / Controversial
    categories: Optional[str]  # 逗号分隔的类别
    refusal: Optional[str]  # Yes / No（仅 response moderation 有）
    raw_output: str  # 模型原始输出
    harmful: bool  # 基于 safety_label 推导
    error: str = ""  # 解析错误信息

    def to_dict(self) -> dict:
        return asdict(self)


class ResultParser:
    """
    解析 Qwen3Guard 模型输出。

    Qwen3Guard 输出格式:
      Safety: Safe|Unsafe|Controversial
      Categories: Violent|Non-violent Illegal Acts|...|None
      Refusal: Yes|No  (仅 response moderation)
    """

    SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
    CATEGORY_PATTERN = re.compile(
        r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|"
        r"PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
        r"Copyright Violation|Jailbreak|None)"
    )
    REFUSAL_PATTERN = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)

    def parse(self, session_id: int, raw_output: str) -> GuardResult:
        """解析单条模型输出"""
        safety_label = None
        categories = None
        refusal = None
        error = ""

        # 提取 Safety label
        safety_match = self.SAFETY_PATTERN.search(raw_output)
        if safety_match:
            safety_label = safety_match.group(1)
        else:
            error = f"无法解析 Safety label"
            logger.warning(f"Session {session_id}: {error}. 原始输出: {raw_output[:200]}")

        # 提取 Categories
        cat_matches = self.CATEGORY_PATTERN.findall(raw_output)
        if cat_matches:
            categories = ", ".join(cat_matches)

        # 提取 Refusal
        refusal_match = self.REFUSAL_PATTERN.search(raw_output)
        if refusal_match:
            refusal = refusal_match.group(1)

        # 推导 harmful
        harmful = (safety_label == "Unsafe") if safety_label else False

        return GuardResult(
            session_id=session_id,
            safety_label=safety_label,
            categories=categories,
            refusal=refusal,
            raw_output=raw_output,
            harmful=harmful,
            error=error,
        )
