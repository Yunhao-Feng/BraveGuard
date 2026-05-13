# Guard 模型轨迹安全评估

## 1. 项目概述

使用 Guard 模型（Qwen3Guard, LlamaGuard 等）对 AI Agent 的执行轨迹进行有害性判断。

### 支持的模型

- **Qwen3Guard-8B / 4B**: 输出结构化的安全评估（Safety/Categories/Refusal）
- **LlamaGuard-8B**: 输出简单的 safe/unsafe 判断

系统会根据模型路径自动识别模型类型。

### 三种实验模式

| 模式 | 名称 | 描述 |
|------|------|------|
| `mode1` | 完整上下文 | 包含 target/comment/jailbreak_method/category + 评估标准 |
| `mode2` | 仅评估标准 | 去掉攻击元信息，仅保留评估标准 + 轨迹 |
| `mode3` | 纯轨迹判断 | 无任何提示，直接判断轨迹是否有害 |

### 项目结构

```
evaluator/
├── base_engine.py          # 引擎抽象基类
├── base_parser.py          # 解析器抽象基类
├── model_engine.py         # 具体引擎实现（Qwen3Guard, LlamaGuard）
├── result_parser.py        # 具体解析器实现
├── factory.py              # 工厂模式（自动创建引擎和解析器）
├── pipeline.py             # 评估流水线
├── config.py               # 配置
├── data_loader.py          # 数据加载
├── prompt_builder.py       # Prompt 构建
└── csv_writer.py           # CSV 输出
```

## 2. 环境准备

### 安装依赖

```bash
pip install vllm>=0.9.0 transformers>=4.51.0 pandas
```

### GPU 要求

- 默认 tensor_parallel=8（8 张 GPU）
- 可通过 `--tp` 参数调整

## 3. 运行评估

### 单模型评估

```bash
python run_eval.py \
    --input exports_v8 \
    --model-paths model_cache/qwen3_guard_8b \
    --mode 3
```

输出：`guard/qwen3_guard_8b_mode3.csv`

### 多模型评估

```bash
python run_eval.py \
    --input exports_v8 \
    --model-paths \
        model_cache/qwen3_guard_8b \
        model_cache/qwen3_guard_4b \
        model_cache/llama3-guard-8B \
    --mode 3
```

输出：
- `guard/qwen3_guard_8b_mode3.csv`
- `guard/qwen3_guard_4b_mode3.csv`
- `guard/llama3-guard-8B_mode3.csv`

### 完整示例

```bash
python run_eval.py \
    --input exports_v8 \
    --model-paths model_cache/qwen3_guard_8b model_cache/llama3-guard-8B \
    --output-dir ./results \
    --mode 3 \
    --dataset data/subset.json \
    --tp 8 \
    --batch-size 32 \
    --max-model-len 32768 \
    --max-new-tokens 128
```

## 4. 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `exports_v8` | 轨迹 JSONL 所在目录 |
| `--model-paths` | `-m` | `model_cache/qwen3_guard_8b` | 模型路径列表（空格分隔，支持多个）|
| `--output-dir` | — | `guard` | 输出目录 |
| `--mode` | — | `3` | 1/2/3 三种实验模式 |
| `--dataset` | `-d` | `data/subset.json` | 数据集路径（mode1 必需） |
| `--tp` | — | `8` | tensor parallel 大小 |
| `--max-model-len` | — | `32768` | 最大上下文长度 |
| `--batch-size` | — | `32` | batch 推理批次大小 |
| `--max-new-tokens` | — | `128` | 生成最大 token 数 |

### 性能调优

- **内存不足**：降低 `--batch-size` 或 `--max-model-len`
- **GPU 数量不同**：调整 `--tp`（如 4 或 2）
- **加速推理**：增加 `--batch-size`（需足够显存）

## 5. 输出格式

### CSV 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | int | 轨迹 ID |
| `harmful` | bool | 是否有害 |
| `safety_label` | str | Safe / Unsafe / Controversial |
| `categories` | str | 风险类别（逗号分隔）|
| `refusal` | str | 是否拒绝（Qwen3Guard 独有）|
| `raw_output` | str | 模型原始输出 |
| `error` | str | 解析错误信息 |

### 模型输出差异

**Qwen3Guard**:
```
Safety: Unsafe
Categories: Non-violent Illegal Acts
Refusal: No
```

**LlamaGuard**:
```
unsafe
S1
```

## 6. 结果分析

```python
import pandas as pd

# 单模型 ASR
df = pd.read_csv('guard/qwen3_guard_8b_mode3.csv')
asr = df['harmful'].mean()
print(f"ASR: {asr:.4f}")

# 多模型对比
for model in ['qwen3_guard_8b', 'qwen3_guard_4b', 'llama3-guard-8B']:
    df = pd.read_csv(f'guard/{model}_mode3.csv')
    print(f"{model}: {df['harmful'].mean():.4f}")
```

## 7. 添加新模型

1. 在 `evaluator/model_engine.py` 中创建新的 Engine 类（继承 `BaseGuardEngine`）
2. 在 `evaluator/result_parser.py` 中创建新的 Parser 类（继承 `BaseResultParser`）
3. 在 `evaluator/factory.py` 中更新 `detect_model_type()` 和 `create_engine_and_parser()`

然后直接使用：
```bash
python run_eval.py --model-paths model_cache/new_model --mode 3
```
