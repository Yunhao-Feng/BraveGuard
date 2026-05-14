"""评估配置数据类"""

from dataclasses import dataclass


@dataclass
class EvalConfig:
    """评估配置"""

    input_dir: str
    model_path: str
    output_path: str | None  # None 时不输出 CSV
    mode: int  # 1, 2, 3
    dataset_path: str = "data/subset.json"
    tensor_parallel: int = 8
    max_model_len: int = 32768
    max_new_tokens: int = 128
    batch_size: int = 32  # batch 推理的批次大小
    model_type: str | None = None  # qwen3 / llama3；None 时从路径自动识别
    prompt_style: str = "response_moderation"  # response_moderation / sft_flat
    annotation_path: str | None = None  # None 时自动尝试 input_dir/results.csv
    enable_thinking: bool | None = False  # Qwen3 推理默认关闭 reasoning/thinking 模板
