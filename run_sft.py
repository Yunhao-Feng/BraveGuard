#!/usr/bin/env python3
"""Prepare and launch LLaMA-Factory SFT for guard models on OpenClaw trajectories."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from sft.data_builder import GuardSFTDataBuilder


def detect_model_type(model_path: str) -> str:
    """Infer guard model family from path without importing vLLM-dependent eval modules."""
    model_path_lower = model_path.lower()
    if "llama" in model_path_lower and "guard" in model_path_lower:
        return "llama3"
    if "qwen" in model_path_lower and "guard" in model_path_lower:
        return "qwen3"
    print(f"[SFT] 无法从模型路径 {model_path!r} 自动识别类型，默认使用 qwen3")
    return "qwen3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 LLaMA-Factory 对 Qwen3Guard/LlamaGuard 做轨迹安全判断 SFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只准备数据和训练配置，不启动训练
  python run_sft.py --input exports --model-path Qwen/Qwen3Guard-Gen-8B --output-dir sft_runs/qwen3_8b --dry-run

  # 启动 LoRA SFT
  python run_sft.py --input exports --model-path Qwen/Qwen3Guard-Gen-8B --model-type qwen3 \
      --output-dir sft_runs/qwen3_8b --template qwen3 --epochs 3 --learning-rate 1e-5

  # LlamaGuard 输出格式（safe/unsafe）
  python run_sft.py --input exports --model-path model_cache/llama3-guard-8B --model-type llama3 \
      --output-dir sft_runs/llama3_guard --template llama3
        """,
    )
    parser.add_argument("--input", "-i", default="exports", help="轨迹 JSONL 文件目录")
    parser.add_argument("--dataset", "-d", default="data/subset.json", help="元信息数据集路径（mode1 使用）")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=3, help="构造 SFT prompt 的实验模式")
    parser.add_argument("--model-path", "-m", required=True, help="基础模型路径或 HuggingFace ID")
    parser.add_argument("--model-type", choices=["auto", "qwen3", "llama3"], default="auto", help="输出格式/模板模型类型")
    parser.add_argument("--output-dir", "-o", required=True, help="SFT 输出目录（adapter/checkpoints/config 都写在这里）")
    parser.add_argument("--dataset-name", default="braveguard_sft", help="LLaMA-Factory dataset 名称")
    parser.add_argument("--annotation-path", help="可选标注 CSV/JSON；默认自动使用轨迹目录下的 results.csv 或唯一 *.csv")
    parser.add_argument("--fallback-label", choices=["safe", "unsafe"], help="仅当某条轨迹缺少标注时使用的兜底标签；默认不兜底并报错")
    parser.add_argument("--category", default="Jailbreak", help="Qwen3Guard Unsafe 样本默认 Categories")
    parser.add_argument("--refusal", choices=["Yes", "No"], default="No", help="Qwen3Guard 默认 Refusal 字段")
    parser.add_argument("--val-size", type=float, default=0.0, help="从训练样本切分验证集的比例，如 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # LLaMA-Factory training arguments
    parser.add_argument("--llamafactory-cli", default="llamafactory-cli", help="LLaMA-Factory CLI 可执行文件")
    parser.add_argument("--template", default=None, help="LLaMA-Factory template；默认 qwen3/llama3")
    parser.add_argument("--finetuning-type", choices=["lora", "full", "freeze"], default="lora")
    parser.add_argument("--stage", default="sft")
    parser.add_argument("--cutoff-len", type=int, default=32768)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target", default="all")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--deepspeed", help="可选 DeepSpeed 配置路径")
    parser.add_argument("--extra-arg", action="append", default=[], help="额外 LLaMA-Factory YAML 参数，格式 key=value，可重复")
    parser.add_argument("--export-dir", help="LoRA 合并导出的完整模型目录；默认 output-dir/merged")
    parser.add_argument("--export-after-train", action="store_true", help="训练成功后调用 llamafactory-cli export 合并 LoRA，便于 run_eval 直接加载")
    parser.add_argument("--dry-run", action="store_true", help="只生成数据和 YAML，不执行训练")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    config_path = output_dir / "train.yaml"
    export_config_path = output_dir / "export.yaml"

    model_type = detect_model_type(args.model_path) if args.model_type == "auto" else args.model_type
    template = args.template or model_type

    builder = GuardSFTDataBuilder(
        input_dir=args.input,
        dataset_path=args.dataset,
        mode=args.mode,
        model_type=model_type,
        fallback_label=args.fallback_label,
        category=args.category,
        refusal=args.refusal,
        annotation_path=args.annotation_path,
        seed=args.seed,
    )
    dataset_files = builder.write_dataset(
        output_dir=str(data_dir),
        dataset_name=args.dataset_name,
        val_size=args.val_size,
    )

    config = build_train_config(args, model_type, template, data_dir)
    export_config = build_export_config(args, template)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(config_path, config)
    if args.finetuning_type == "lora":
        write_yaml(export_config_path, export_config)

    print(f"[SFT] 模型类型: {model_type}")
    print(f"[SFT] 数据目录: {dataset_files['dataset_dir']}")
    print(f"[SFT] 训练配置: {config_path}")
    if args.finetuning_type == "lora":
        print(f"[SFT] LoRA 合并导出配置: {export_config_path}")

    command = [args.llamafactory_cli, "train", str(config_path)]
    if args.dry_run:
        print("[SFT] dry-run 已启用，不启动训练。")
        print("[SFT] 可手动执行:", " ".join(command))
        if args.finetuning_type == "lora":
            print("[SFT] 训练后可合并 LoRA:", " ".join([args.llamafactory_cli, "export", str(export_config_path)]))
        return 0

    if shutil.which(args.llamafactory_cli) is None:
        print(
            f"找不到 {args.llamafactory_cli}。请先安装 LLaMA-Factory，或用 --llamafactory-cli 指定路径。",
            file=sys.stderr,
        )
        return 127

    print("[SFT] 启动训练:", " ".join(command))
    train_result = subprocess.run(command, check=False).returncode
    if train_result != 0 or not args.export_after_train or args.finetuning_type != "lora":
        return train_result

    export_command = [args.llamafactory_cli, "export", str(export_config_path)]
    print("[SFT] 合并 LoRA:", " ".join(export_command))
    return subprocess.run(export_command, check=False).returncode


def build_train_config(args: argparse.Namespace, model_type: str, template: str, data_dir: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "model_name_or_path": args.model_path,
        "stage": args.stage,
        "do_train": True,
        "finetuning_type": args.finetuning_type,
        "template": template,
        "dataset": args.dataset_name,
        "dataset_dir": str(data_dir),
        "cutoff_len": args.cutoff_len,
        "output_dir": str(Path(args.output_dir) / "adapter"),
        "overwrite_cache": True,
        "overwrite_output_dir": True,
        "preprocessing_num_workers": 8,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "plot_loss": True,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    if args.val_size > 0:
        config.update({"eval_dataset": f"{args.dataset_name}_eval", "val_size": 0.0, "do_eval": True})
    if args.finetuning_type == "lora":
        config.update(
            {
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target": args.lora_target,
            }
        )
    if args.deepspeed:
        config["deepspeed"] = args.deepspeed

    for item in args.extra_arg:
        if "=" not in item:
            raise ValueError(f"--extra-arg 必须是 key=value，收到: {item}")
        key, value = item.split("=", 1)
        config[key] = parse_scalar(value)

    return config


def build_export_config(args: argparse.Namespace, template: str) -> Dict[str, Any]:
    export_dir = args.export_dir or str(Path(args.output_dir) / "merged")
    return {
        "model_name_or_path": args.model_path,
        "adapter_name_or_path": str(Path(args.output_dir) / "adapter"),
        "template": template,
        "finetuning_type": "lora",
        "export_dir": export_dir,
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def write_yaml(path: Path, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            f.write(f"{key}: {format_yaml_value(value)}\n")


def format_yaml_value(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


if __name__ == "__main__":
    raise SystemExit(main())
