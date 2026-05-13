#!/usr/bin/env python3
"""
Guard 模型轨迹安全评估入口脚本

支持多种 Guard 模型（Qwen3Guard, LlamaGuard 等）对 AI Agent 执行轨迹进行有害性判断。
支持三种实验模式：
  - mode1: 完整上下文（含 target/comment/jailbreak_method/category + 评估标准）
  - mode2: 仅评估标准（不含攻击元信息）
  - mode3: 纯轨迹判断（无任何提示，直接判断轨迹是否有害）

支持多模型评估：可同时对多个模型进行评估并输出各自的 CSV 结果。
"""

import argparse
import sys
from pathlib import Path
from typing import List

from evaluator.config import EvalConfig
from evaluator.pipeline import EvalPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guard 模型轨迹安全评估工具（支持 Qwen3Guard, LlamaGuard 等）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型评估
  python run_eval.py --input ./exports_v8 --model-paths model_cache/qwen3_guard_8b --mode 3

  # 多模型评估（同时评估 3 个模型）
  python run_eval.py --input ./exports_v8 \\
      --model-paths model_cache/qwen3_guard_8b model_cache/qwen3_guard_4b model_cache/llama3-guard-8B \\
      --mode 3

  # 指定输出目录
  python run_eval.py --input ./exports_v8 \\
      --model-paths model_cache/qwen3_guard_8b model_cache/llama3-guard-8B \\
      --output-dir ./results \\
      --mode 3
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
        "--model-paths", "-m",
        type=str,
        nargs="+",
        required=False,
        default=["model_cache/llama3-guard-8B"],
        help="模型路径列表，支持多个模型（空格分隔）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="guard",
        help="输出目录（默认: guard/），每个模型的结果保存为单独的 CSV 文件",
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
        "--max-new-tokens",
        type=int,
        default=128,
        help="生成最大 token 数（默认 128）",
    )

    args = parser.parse_args()
    return args


def extract_model_name(model_path: str) -> str:
    """
    从模型路径提取简短的模型名称。

    例如：
      model_cache/qwen3_guard_8b -> qwen3_guard_8b
      /path/to/llama3-guard-8B -> llama3-guard-8B
    """
    return Path(model_path).name


def generate_output_path(model_path: str, output_dir: str, mode: int) -> str:
    """
    为模型生成输出 CSV 路径。

    Args:
        model_path: 模型路径
        output_dir: 输出目录
        mode: 实验模式

    Returns:
        输出 CSV 文件路径
    """
    model_name = extract_model_name(model_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    return str(output_dir_path / f"{model_name}_mode{mode}.csv")


def run_single_model_eval(
    model_path: str,
    input_dir: str,
    output_path: str,
    mode: int,
    dataset_path: str,
    tensor_parallel: int,
    max_model_len: int,
    max_new_tokens: int,
    batch_size: int,
):
    """
    运行单个模型的评估。

    Args:
        model_path: 模型路径
        input_dir: 输入目录
        output_path: 输出 CSV 路径
        mode: 实验模式
        dataset_path: 数据集路径
        tensor_parallel: Tensor parallel 大小
        max_model_len: 最大模型长度
        max_new_tokens: 最大生成 token 数
        batch_size: Batch 大小
    """
    config = EvalConfig(
        input_dir=input_dir,
        model_path=model_path,
        output_path=output_path,
        mode=mode,
        dataset_path=dataset_path,
        tensor_parallel=tensor_parallel,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    model_name = extract_model_name(model_path)
    print("=" * 80)
    print(f"[Guard Eval] 模型: {model_name}")
    print(f"[Guard Eval] 模式: mode{config.mode}")
    print(f"[Guard Eval] 输入目录: {config.input_dir}")
    print(f"[Guard Eval] 模型路径: {config.model_path}")
    print(f"[Guard Eval] 输出路径: {config.output_path or '(不输出 CSV)'}")
    print(f"[Guard Eval] Tensor Parallel: {config.tensor_parallel}")
    print(f"[Guard Eval] Batch Size: {config.batch_size}")
    print("=" * 80)
    print()

    pipeline = EvalPipeline(config)
    pipeline.run()
    print()


def main():
    args = parse_args()

    # 多模型评估
    model_paths = args.model_paths
    num_models = len(model_paths)

    print(f"\n{'=' * 80}")
    print(f"开始多模型评估，共 {num_models} 个模型")
    print(f"{'=' * 80}\n")

    for idx, model_path in enumerate(model_paths, 1):
        print(f"\n>>> 正在评估模型 [{idx}/{num_models}]: {model_path}\n")

        # 生成输出路径
        output_path = generate_output_path(model_path, args.output_dir, args.mode)

        # 运行评估
        try:
            run_single_model_eval(
                model_path=model_path,
                input_dir=args.input,
                output_path=output_path,
                mode=args.mode,
                dataset_path=args.dataset,
                tensor_parallel=args.tensor_parallel,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )
            print(f"✓ 模型 {extract_model_name(model_path)} 评估完成\n")
        except Exception as e:
            print(f"✗ 模型 {extract_model_name(model_path)} 评估失败: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"所有模型评估完成！结果保存在: {args.output_dir}/")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
