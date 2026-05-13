"""模型推理引擎抽象基类"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseGuardEngine(ABC):
    """
    Guard 模型推理引擎抽象基类。

    定义所有 Guard 模型引擎的统一接口，具体实现由子类完成。
    """

    @abstractmethod
    def initialize(self):
        """初始化模型和 tokenizer"""
        pass

    @abstractmethod
    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        构建 prompt。

        Args:
            messages: 对话消息列表，格式为 [{"role": "user", "content": "..."}, ...]

        Returns:
            格式化后的 prompt 字符串
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        session_ids: List[int] = None,
        max_input_tokens: int = 32000,
    ) -> List[str]:
        """
        批量推理。

        Args:
            prompts: 格式化后的 prompt 列表
            session_ids: 对应的 session_id 列表（用于日志）
            max_input_tokens: 最大输入 token 数

        Returns:
            模型输出文本列表，顺序与输入一致
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """返回模型类型标识（如 'qwen3', 'llama3'）"""
        pass
