"""结果解析模块：解析不同 Guard 模型的输出"""

import re
import logging
from typing import Optional

from .base_parser import BaseResultParser, GuardResult

logger = logging.getLogger(__name__)


class Qwen3ResultParser(BaseResultParser):
    """
    解析 Qwen3Guard 模型输出。

    Qwen3Guard 输出格式:
      Safety: Safe|Unsafe|Controversial
      Categories: Violent|Non-violent Illegal Acts|...|None
      Refusal: Yes|No  (仅 response moderation)
    """

    SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
    SHORT_LABEL_PATTERN = re.compile(r"^\s*(safe|unsafe|controversial)\b", re.IGNORECASE)
    CATEGORY_PATTERN = re.compile(
        r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|"
        r"PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
        r"Copyright Violation|Jailbreak|None)",
        re.IGNORECASE,
    )
    REFUSAL_PATTERN = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)
    CATEGORY_LINE_PATTERN = re.compile(r"Categories:\s*([^\n]+)", re.IGNORECASE)

    def parse(self, session_id: int, raw_output: str) -> GuardResult:
        """解析单条模型输出"""
        safety_label = None
        categories = None
        refusal = None
        error = ""

        # 提取 Safety label
        safety_match = self.SAFETY_PATTERN.search(raw_output)
        if safety_match:
            safety_label = safety_match.group(1).strip().capitalize()
        else:
            short_match = self.SHORT_LABEL_PATTERN.search(raw_output)
            if short_match:
                safety_label = short_match.group(1).strip().capitalize()
            else:
                error = f"无法解析 Safety label"
                logger.warning(f"Session {session_id}: {error}. 原始输出: {raw_output[:200]}")

        # 提取 Categories：优先取整行，兼容 CSV 中的自定义 category（如 System_RCE）。
        category_line = self.CATEGORY_LINE_PATTERN.search(raw_output)
        if category_line:
            categories = category_line.group(1).strip()
        else:
            cat_matches = self.CATEGORY_PATTERN.findall(raw_output)
            if cat_matches:
                categories = ", ".join(cat_matches)

        # 提取 Refusal
        refusal_match = self.REFUSAL_PATTERN.search(raw_output)
        if refusal_match:
            refusal = refusal_match.group(1).strip().capitalize()

        # 推导 harmful
        harmful = (safety_label.upper() == "UNSAFE") if safety_label else False

        return GuardResult(
            session_id=session_id,
            safety_label=safety_label,
            categories=categories,
            refusal=refusal,
            raw_output=raw_output,
            harmful=harmful,
            error=error,
        )


class LlamaResultParser(BaseResultParser):
    """
    解析 LlamaGuard 模型输出。

    LlamaGuard 输出格式简单：
      - "safe": 安全
      - "unsafe": 不安全
      - 可能带有额外说明（如分类），但主要判断标准是 safe/unsafe
    """

    def parse(self, session_id: int, raw_output: str) -> GuardResult:
        """解析 LlamaGuard 输出"""
        safety_label = None
        categories = None
        error = ""

        # 转为小写进行判断
        output_lower = raw_output.lower().strip()

        # 判断是否包含 safe 或 unsafe
        if "unsafe" in output_lower:
            safety_label = "Unsafe"
            harmful = True
        elif "safe" in output_lower:
            safety_label = "Safe"
            harmful = False
        else:
            # 无法识别
            error = "无法解析 LlamaGuard 输出（期望 'safe' 或 'unsafe'）"
            logger.warning(f"Session {session_id}: {error}. 原始输出: {raw_output[:200]}")
            harmful = False

        # LlamaGuard 可能输出类别信息（如 S1, S2 等），尝试提取
        # 格式可能是：unsafe\nS1
        category_pattern = re.compile(r"\b(S\d+|O\d+)\b")
        cat_matches = category_pattern.findall(raw_output)
        if cat_matches:
            categories = ", ".join(cat_matches)

        return GuardResult(
            session_id=session_id,
            safety_label=safety_label,
            categories=categories,
            refusal=None,  # LlamaGuard 不输出 refusal
            raw_output=raw_output,
            harmful=harmful,
            error=error,
        )
