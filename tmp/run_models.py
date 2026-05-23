"""批量遍历模型列表，修改 config/config_local.json 的 agent model 并运行 local_runner.py"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config_local.json"

# 模型列表 - 按需修改
MODELS = [
    "gpt-5.5-0424-global",
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    "qwen3-235b-a22b-instruct-2507",
    "kimi-k2.5"
]


def update_model(model_name: str):
    """修改配置文件中的 agent model"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["agent"]["model"] = model_name

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[CONFIG] agent.model -> {model_name}")


def run_local_runner(dataset: str):
    """运行 local_runner.py，返回是否成功"""
    cmd = [sys.executable, "local_runner.py", "--dataset", dataset]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def move_exports(model_name: str, dataset: str):
    """将 exports_local 移动到 tmp/workspace/data/{dataset}/{model_name}/ 避免被覆盖"""
    src = PROJECT_ROOT / "exports_local"
    dst = PROJECT_ROOT / "tmp" / "workspace" / "data" / dataset / model_name

    if not src.exists():
        print(f"[WARN] exports_local 不存在，跳过移动")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)

    shutil.move(str(src), str(dst))
    print(f"[MOVE] exports_local -> {dst}")


DATASET_CHOICES = ["agenthazard_strongest", "atbench_trans"]


def parse_args():
    parser = argparse.ArgumentParser(description="批量遍历模型列表运行评测")
    parser.add_argument(
        "--dataset", "-d",
        choices=DATASET_CHOICES,
        default="agenthazard_strongest",
        help="选择数据集（默认: agenthazard_strongest）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset

    print("=" * 60)
    print(f"批量模型评测 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置文件: {CONFIG_PATH}")
    print(f"数据集: {dataset}")
    print(f"模型数量: {len(MODELS)}")
    print("=" * 60)

    results = []

    for i, model in enumerate(MODELS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(MODELS)}] 运行模型: {model}")
        print(f"{'='*60}")

        # 1. 修改配置
        update_model(model)

        # 2. 运行 local_runner
        start_time = datetime.now()
        success = run_local_runner(dataset)
        elapsed = datetime.now() - start_time

        # 3. 移动结果目录，防止下一轮覆盖
        move_exports(model, dataset)

        status = "SUCCESS" if success else "FAILED"
        results.append((model, status, str(elapsed)))
        print(f"\n[{status}] {model} - 耗时: {elapsed}")

    # 汇总
    print("\n" + "=" * 60)
    print("运行汇总:")
    print("=" * 60)
    for model, status, elapsed in results:
        print(f"  {status:8s} | {elapsed:>15s} | {model}")


if __name__ == "__main__":
    main()
