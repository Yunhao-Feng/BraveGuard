#!/usr/bin/env python3
"""
Qwen3Guard 轨迹安全评估入口脚本

使用 vLLM 部署 Qwen3Guard-Gen-8B 模型，对 AI Agent 执行轨迹进行有害性判断。
支持三种实验模式：
  - mode1: 完整上下文（含 target/comment/jailbreak_method/category + 评估标准）
  - mode2: 仅评估标准（不含攻击元信息）
  - mode3: 纯轨迹判断（无任何提示，直接判断轨迹是否有害）
"""

import argparse
import sys
from pathlib import Path

from evaluator.config import EvalConfig
from evaluator.pipeline import EvalPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3Guard 轨迹安全评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Mode 1: 完整上下文评估
  python run_eval.py --input ./exports_v8 --model-path /path/to/Qwen3Guard-Gen-8B --output ./guard/output_mode1.csv --mode 1

  # Mode 2: 仅评估标准（去掉攻击元信息）
  python run_eval.py --input ./exports_v8 --model-path /path/to/Qwen3Guard-Gen-8B --output ./guard/output_mode2.csv --mode 2

  # Mode 3: 纯轨迹判断
  python run_eval.py --input ./exports_v8 --model-path /path/to/Qwen3Guard-Gen-8B --output ./guard/output_mode3.csv --mode 3
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=False,
        default="exports_v8",
        help="轨迹 JSONL 文件所在文件夹路径（如 exports_v8）",
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=False,
        default="qwen3_guard_8b",
        help="Qwen3Guard-Gen-8B 模型路径（本地路径或 HuggingFace ID）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出 CSV 文件路径（默认: guard/output_mode{N}.csv，设为 None 则不输出 CSV）",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3],
        required=False,
        default= 3,
        help="实验模式: 1=完整上下文, 2=仅评估标准, 3=纯轨迹",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="data/subset.json",
        help="数据集路径，用于获取 target/comment 等元信息（mode1 必需）",
    )
    parser.add_argument(
        "--tp", "--tensor-parallel",
        type=int,
        default=8,
        dest="tensor_parallel",
        help="vLLM tensor parallel 大小（默认 8，对应 8 张 GPU）",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM 最大模型长度（默认 32768）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch 推理的批次大小（默认 32）",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=64,
        help="已废弃（保留兼容性）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="生成最大 token 数（默认 128）",
    )

    args = parser.parse_args()

    # 默认输出路径
    # if args.output is None:
    #     args.output = f"guard/output_mode{args.mode}.csv"

    return args


def main():
    args = parse_args()

    config = EvalConfig(
        input_dir=args.input,
        model_path=args.model_path,
        output_path=args.output,
        mode=args.mode,
        dataset_path=args.dataset,
        tensor_parallel=args.tensor_parallel,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
    )

    print(f"[Guard Eval] 模式: mode{config.mode}")
    print(f"[Guard Eval] 输入目录: {config.input_dir}")
    print(f"[Guard Eval] 模型路径: {config.model_path}")
    print(f"[Guard Eval] 输出路径: {config.output_path or '(不输出 CSV)'}")
    print(f"[Guard Eval] Tensor Parallel: {config.tensor_parallel}")
    print(f"[Guard Eval] Batch Size: {config.batch_size}")
    print()

    pipeline = EvalPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
