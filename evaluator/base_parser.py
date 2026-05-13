"""结果解析器抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional


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


class BaseResultParser(ABC):
    """
    结果解析器抽象基类。

    不同模型的输出格式不同，需要不同的解析器。
    """

    @abstractmethod
    def parse(self, session_id: int, raw_output: str) -> GuardResult:
        """
        解析模型输出。

        Args:
            session_id: 会话 ID
            raw_output: 模型原始输出文本

        Returns:
            解析后的结果对象
        """
        pass
