#!/usr/bin/env python3
"""
绘制三个模型的训练损失曲线图
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

# 设置学术论文风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 15

# 使用 seaborn 样式
sns.set_style("whitegrid")
sns.set_palette("deep")


def load_training_logs(log_file: str) -> pd.DataFrame:
    """
    加载训练日志，只保留包含 loss 的记录（排除 eval_loss）

    Args:
        log_file: trainer_log.jsonl 文件路径

    Returns:
        包含 step 和 loss 的 DataFrame
    """
    steps = []
    losses = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # 只保留有 loss 字段且没有 eval_loss 字段的记录
                if 'loss' in data and 'eval_loss' not in data:
                    steps.append(data['current_steps'])
                    losses.append(data['loss'])
            except json.JSONDecodeError:
                continue

    return pd.DataFrame({'step': steps, 'loss': losses})


def smooth_curve(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    使用移动平均平滑曲线

    Args:
        data: 原始数据
        window_size: 窗口大小

    Returns:
        平滑后的数据
    """
    return uniform_filter1d(data, size=window_size, mode='nearest')


def plot_single_model_loss(
    df: pd.DataFrame,
    model_name: str,
    ax: plt.Axes,
    color: str,
    smooth_window: int = 100
):
    """
    在给定的 axes 上绘制单个模型的损失曲线

    Args:
        df: 包含 step 和 loss 的 DataFrame
        model_name: 模型名称（用于图例）
        ax: matplotlib axes
        color: 曲线颜色
        smooth_window: 平滑窗口大小
    """
    steps = df['step'].values
    losses = df['loss'].values

    # 原始曲线（半透明）
    ax.plot(
        steps, losses,
        alpha=0.2,
        linewidth=0.8,
        color=color,
        label='Raw'
    )

    # 平滑曲线
    smoothed_losses = smooth_curve(losses, window_size=smooth_window)
    ax.plot(
        steps, smoothed_losses,
        linewidth=2.5,
        color=color,
        label=model_name,
        zorder=3
    )

    # 设置坐标轴
    ax.set_xlabel('Training Steps', fontweight='bold', fontsize=12)
    ax.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
    ax.set_title(f'{model_name} - Training Loss Curve', fontweight='bold', fontsize=14, pad=15)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 图例
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=11)

    # Y轴从0开始
    ax.set_ylim(bottom=0, top=max(losses) * 1.1)

    # 添加统计信息
    final_loss = smoothed_losses[-1]
    min_loss = smoothed_losses.min()
    ax.text(
        0.02, 0.98,
        f'Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
        fontsize=10
    )


def main():
    """主函数：绘制三个模型的训练损失曲线"""

    # 模型配置
    models = [
        {
            'path': 'sft_runs/llama3_guard_8b',
            'name': 'llama3_guard_8b',
            'color': '#E74C3C'  # 红色
        },
        {
            'path': 'sft_runs/qwen3_guard_4b',
            'name': 'qwen3_guard_4b',
            'color': '#3498DB'  # 蓝色
        },
        {
            'path': 'sft_runs/qwen3_guard_8b',
            'name': 'qwen3_guard_8b',
            'color': '#27AE60'  # 绿色
        }
    ]

    # 创建图形：3行1列
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    print("=" * 80)
    print("绘制训练损失曲线")
    print("=" * 80)

    # 为每个模型绘制损失曲线
    for idx, model_config in enumerate(models):
        model_path = model_config['path']
        model_name = model_config['name']
        color = model_config['color']

        log_file = f"{model_path}/adapter/trainer_log.jsonl"

        print(f"\n处理模型: {model_name}")
        print(f"  日志文件: {log_file}")

        # 检查文件是否存在
        if not Path(log_file).exists():
            print(f"  ⚠️  警告: 日志文件不存在，跳过")
            continue

        # 加载训练日志
        df = load_training_logs(log_file)
        print(f"  加载了 {len(df)} 条训练记录")

        if len(df) == 0:
            print(f"  ⚠️  警告: 没有找到训练损失记录，跳过")
            continue

        # 绘制损失曲线
        plot_single_model_loss(
            df=df,
            model_name=model_name,
            ax=axes[idx],
            color=color,
            smooth_window=100
        )

        print(f"  ✓ 已绘制 {model_name} 的损失曲线")

    # 调整布局
    plt.tight_layout(pad=3.0)

    # 保存图形
    output_pdf = "training_loss_curves.pdf"
    output_png = "training_loss_curves.png"

    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n✓ PDF 图表已保存至: {output_pdf}")

    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ PNG 预览已保存至: {output_png}")

    print("\n" + "=" * 80)
    print("绘制完成！")
    print("=" * 80 + "\n")

    plt.close()


if __name__ == "__main__":
    main()
