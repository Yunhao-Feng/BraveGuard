"""模型推理引擎：使用 vLLM 进行批量推理"""

import logging
from typing import Dict, List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .config import EvalConfig
from .base_engine import BaseGuardEngine

logger = logging.getLogger(__name__)


class Qwen3GuardEngine(BaseGuardEngine):
    """
    基于 vLLM 的 Qwen3Guard 推理引擎。

    使用 vLLM 的离线批量推理（offline batch inference）能力，
    充分利用多 GPU 的 tensor parallelism 加速推理。
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer = None
        self.llm = None
        self.sampling_params = None

    def initialize(self):
        """初始化模型和 tokenizer"""
        logger.info(f"正在加载模型: {self.config.model_path}")
        logger.info(f"Tensor Parallel: {self.config.tensor_parallel}")
        logger.info(f"Max Model Len: {self.config.max_model_len}")

        # 加载 tokenizer（用于 apply_chat_template）
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # 初始化 vLLM 引擎
        self.llm = LLM(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
            dtype="auto",
        )

        # 设置采样参数
        self.sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )

        logger.info("模型加载完成")

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        使用 tokenizer 的 chat template 构建 prompt。

        Qwen3Guard 使用标准的 chat template 格式。
        如果 prompt 超过最大长度，会自动截断。
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # 添加生成 prompt 后缀
            enable_thinking=False,  # Qwen3 reasoning 模板下显式关闭 thinking/no_think 分支
        )

        # 检查并截断过长的 prompt
        input_ids = self.tokenizer.encode(text)
        original_len = len(input_ids)

        # 如果超过最大长度，保留后面的 tokens（保留最新的上下文）
        if original_len > self.config.max_model_len:
            logger.warning(
                f"Prompt 长度 {original_len} 超过最大长度 {self.config.max_model_len}，"
                f"截断到 {self.config.max_model_len} tokens"
            )
            input_ids = input_ids[-self.config.max_model_len:]
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        return text

    def batch_generate(
        self,
        prompts: List[str],
        session_ids: List[int] = None,
        max_input_tokens: int = 32000,
    ) -> List[str]:
        """
        批量 prompt 推理（串行调用 vLLM）。

        在调用 vLLM 前，对每条 prompt 做 token 长度检查：
        - 如果超过 max_input_tokens，截断到该长度（保留最后的 tokens）
        - 记录日志

        Args:
            prompts: 格式化后的 prompt 列表
            session_ids: 对应的 session_id 列表（用于日志）
            max_input_tokens: 最大输入 token 数（默认 32000，给输出留余量）

        Returns:
            outputs: 模型输出文本列表，顺序与输入一致
        """
        if session_ids is None:
            session_ids = list(range(len(prompts)))

        # 检查并截断过长的 prompt
        processed_prompts = []
        for idx, (prompt, session_id) in enumerate(zip(prompts, session_ids)):
            # tokenize 并检查长度
            input_ids = self.tokenizer.encode(prompt)
            original_len = len(input_ids)

            if original_len > max_input_tokens:
                logger.warning(
                    f"Session {session_id}: prompt 长度 {original_len} tokens "
                    f"超过限制 {max_input_tokens}，截断保留最后 {max_input_tokens} tokens"
                )
                # 保留最后的 tokens（保留最新的上下文）
                input_ids = input_ids[-max_input_tokens:]
                prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)

            processed_prompts.append(prompt)

        # 批量推理（一次性调用 vLLM）
        logger.info(f"开始批量推理，batch_size={len(processed_prompts)}")
        outputs = self.llm.generate(processed_prompts, self.sampling_params)

        # 提取生成文本
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)

        return results

    @property
    def model_type(self) -> str:
        """返回模型类型"""
        return "qwen3"


class LlamaGuardEngine(BaseGuardEngine):
    """
    基于 vLLM 的 LlamaGuard 推理引擎。

    LlamaGuard 输出格式简单，只有 "safe" 或 "unsafe"。
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer = None
        self.llm = None
        self.sampling_params = None

    def initialize(self):
        """初始化模型和 tokenizer"""
        logger.info(f"正在加载 LlamaGuard 模型: {self.config.model_path}")
        logger.info(f"Tensor Parallel: {self.config.tensor_parallel}")
        logger.info(f"Max Model Len: {self.config.max_model_len}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # 初始化 vLLM 引擎
        self.llm = LLM(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
            dtype="auto",
        )

        # 设置采样参数
        self.sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )

        logger.info("LlamaGuard 模型加载完成")

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        使用 tokenizer 的 chat template 构建 prompt。

        LlamaGuard 使用标准的 chat template 格式。
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 检查并截断过长的 prompt
        input_ids = self.tokenizer.encode(text)
        original_len = len(input_ids)

        if original_len > self.config.max_model_len:
            logger.warning(
                f"Prompt 长度 {original_len} 超过最大长度 {self.config.max_model_len}，"
                f"截断到 {self.config.max_model_len} tokens"
            )
            input_ids = input_ids[-self.config.max_model_len:]
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        return text

    def batch_generate(
        self,
        prompts: List[str],
        session_ids: List[int] = None,
        max_input_tokens: int = 32000,
    ) -> List[str]:
        """
        批量 prompt 推理。

        与 Qwen3GuardEngine 实现类似，但输出格式更简单。
        """
        if session_ids is None:
            session_ids = list(range(len(prompts)))

        # 检查并截断过长的 prompt
        processed_prompts = []
        for idx, (prompt, session_id) in enumerate(zip(prompts, session_ids)):
            input_ids = self.tokenizer.encode(prompt)
            original_len = len(input_ids)

            if original_len > max_input_tokens:
                logger.warning(
                    f"Session {session_id}: prompt 长度 {original_len} tokens "
                    f"超过限制 {max_input_tokens}，截断保留最后 {max_input_tokens} tokens"
                )
                input_ids = input_ids[-max_input_tokens:]
                prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)

            processed_prompts.append(prompt)

        # 批量推理
        logger.info(f"开始批量推理（LlamaGuard），batch_size={len(processed_prompts)}")
        outputs = self.llm.generate(processed_prompts, self.sampling_params)

        # 提取生成文本
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)

        return results

    @property
    def model_type(self) -> str:
        """返回模型类型"""
        return "llama3"
