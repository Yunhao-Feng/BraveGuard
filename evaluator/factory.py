"""工厂模块：根据模型类型自动创建引擎和解析器"""

import logging
from typing import Tuple

from .config import EvalConfig
from .base_engine import BaseGuardEngine
from .base_parser import BaseResultParser
from .model_engine import Qwen3GuardEngine, LlamaGuardEngine
from .result_parser import Qwen3ResultParser, LlamaResultParser

logger = logging.getLogger(__name__)


def detect_model_type(model_path: str) -> str:
    """
    根据模型路径自动检测模型类型。

    Args:
        model_path: 模型路径（本地路径或 HuggingFace ID）

    Returns:
        模型类型字符串：'qwen3' 或 'llama3'
    """
    model_path_lower = model_path.lower()

    if "llama" in model_path_lower and "guard" in model_path_lower:
        return "llama3"
    elif "qwen" in model_path_lower and "guard" in model_path_lower:
        return "qwen3"
    else:
        # 默认使用 qwen3（向后兼容）
        logger.warning(
            f"无法从路径 '{model_path}' 自动识别模型类型，默认使用 qwen3"
        )
        return "qwen3"


def create_engine_and_parser(
    config: EvalConfig,
) -> Tuple[BaseGuardEngine, BaseResultParser]:
    """
    工厂函数：根据配置创建对应的引擎和解析器。

    Args:
        config: 评估配置

    Returns:
        (engine, parser) 元组
    """
    model_type = config.model_type or detect_model_type(config.model_path)
    logger.info(f"检测到模型类型: {model_type}")

    if model_type == "llama3":
        engine = LlamaGuardEngine(config)
        parser = LlamaResultParser()
    elif model_type == "qwen3":
        engine = Qwen3GuardEngine(config)
        parser = Qwen3ResultParser()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return engine, parser
