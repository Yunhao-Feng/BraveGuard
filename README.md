<div align="center">
  <img src="logo.png" alt="BraveGuard Logo" width="200"/>

  # BraveGuard

  **A Comprehensive Framework for Evaluating Guard Models on AI Agent Trajectory Safety**

  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![arXiv](https://img.shields.io/badge/arXiv-2024.EMNLP-b31b1b.svg)](EMNLP_26.pdf)

  [Documentation](#documentation) • [Quick Start](#quick-start) • [Benchmarks](#benchmarks) • [Citation](#citation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Trajectory Generation](#trajectory-generation)
  - [Guard Model Evaluation](#guard-model-evaluation)
  - [Supervised Fine-Tuning](#supervised-fine-tuning)
- [Supported Models](#supported-models)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

**BraveGuard** is a comprehensive research framework designed to evaluate and improve the safety assessment capabilities of guard models on AI agent execution trajectories. As AI agents become increasingly autonomous and capable, ensuring their actions remain safe and aligned with human values is critical. BraveGuard provides tools for:

- **Trajectory Safety Dataset Construction**: Generate diverse agent execution traces with safety annotations
- **Guard Model Evaluation**: Assess how well safety models detect harmful agent behaviors
- **Fine-tuning Pipeline**: Improve guard models through supervised fine-tuning on trajectory data
- **Comprehensive Benchmarking**: Support for multiple guard models (Qwen3Guard, LlamaGuard, etc.)

This framework supports research at the intersection of AI safety, agent systems, and large language model alignment.

---

## ✨ Features

### 🔍 **Multi-Model Support**
- **Qwen3Guard** (4B/8B): Binary safety classification
- **LlamaGuard** (8B): Policy-based content moderation
- **Extensible Architecture**: Easy integration of custom guard models

### 🧪 **Flexible Evaluation Modes**
| Mode | Description | Use Case |
|------|-------------|----------|
| **Mode 1** | Full context with attack metadata | Research on jailbreak methods |
| **Mode 2** | Evaluation criteria only | Realistic deployment scenarios |
| **Mode 3** | Pure trajectory judgment | Zero-shot safety assessment |

### 🚀 **Production-Ready Pipeline**
- Batch inference with vLLM for high throughput
- Multi-GPU tensor parallelism support
- Comprehensive CSV output with accuracy metrics
- Automatic annotation alignment

### 🎓 **Fine-Tuning Capabilities**
- Integration with LLaMA-Factory
- LoRA and full fine-tuning support
- Automatic train/validation split
- Label balancing strategies

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU(s) (8x GPUs recommended for tensor parallelism)
- 50GB+ disk space for model caches

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BraveGuard.git
cd BraveGuard

# Create virtual environment
conda env create -f environment.yml
conda activate braveguard

# Install dependencies
pip install vllm>=0.9.0 transformers>=4.51.0 pandas openai rich pydantic
```

### Model Downloads

```bash
# Download guard models
python download.py --models qwen3-guard-8b llama3-guard-8b

# Or manually download from Hugging Face
# Place models in model_cache/ directory
```

---

## 🚀 Quick Start

### 1. Generate Agent Trajectories

```bash
python generate.py \
    --dataset data/subset.json \
    --output exports/trajectories \
    --config config/config.json
```

### 2. Evaluate Guard Models

```bash
python run_eval.py \
    --input exports/trajectories \
    --model-paths model_cache/qwen3_guard_8b \
    --mode 3 \
    --output-dir results
```

### 3. Analyze Results

```python
import pandas as pd

# Load evaluation results
df = pd.read_csv('results/qwen3_guard_8b_mode3.csv')

# Calculate Attack Success Rate (ASR)
asr = df['harmful'].mean()
print(f"Attack Success Rate: {asr:.2%}")

# Calculate accuracy (if ground truth available)
accuracy = df['correct'].mean()
print(f"Guard Model Accuracy: {accuracy:.2%}")
```

---

## 📁 Project Structure

```
BraveGuard/
├── evaluator/               # Core evaluation engine
│   ├── base_engine.py      # Abstract base classes
│   ├── model_engine.py     # Guard model implementations
│   ├── result_parser.py    # Output parsing logic
│   ├── pipeline.py         # Evaluation orchestration
│   ├── prompt_builder.py   # Prompt construction
│   └── factory.py          # Model factory pattern
├── rock/                    # OpenClaw integration
│   └── openclaw.py         # Agent execution wrapper
├── utils/                   # Utility functions
│   ├── judge.py            # LLM-based judging
│   ├── key_pool.py         # API key management
│   └── config_*.py         # Configuration helpers
├── data/                    # Datasets
│   ├── subset.json         # Sample dataset
│   ├── atbench500.json     # ATBench benchmark
│   └── asse-safety.json    # ASSE-Safety dataset
├── config/                  # Configuration files
│   ├── config.json         # Main configuration
│   └── llm_judge.yaml      # LLM judge settings
├── LlamaFactory/           # Fine-tuning framework
├── run_eval.py             # Evaluation entry point
├── run_sft.py              # Fine-tuning entry point
├── generate.py             # Trajectory generation
└── README.md               # This file
```

---

## 📖 Usage

### Trajectory Generation

BraveGuard supports multiple trajectory generation backends:

#### Using OpenClaw (Recommended)

```bash
python generate.py \
    --dataset data/subset.json \
    --backend openclaw \
    --agent-model gpt-4-turbo \
    --output exports/openclaw_traces \
    --max-steps 10
```

#### Local Agent Execution

```bash
python local_runner.py \
    --dataset data/subset.json \
    --output exports/local_traces \
    --docker-image braveguard/agent:latest
```

**Output Format**: Each trajectory is saved as `session_item-{id}.jsonl` containing:
- Initial query and decomposed steps
- Agent thoughts and actions
- Tool execution results
- Final outcome

---

### Guard Model Evaluation

#### Single Model Evaluation

```bash
python run_eval.py \
    --input exports/trajectories \
    --model-paths model_cache/qwen3_guard_8b \
    --mode 3 \
    --batch-size 32 \
    --tp 8
```

**Output**: `results/qwen3_guard_8b_mode3.csv`

#### Multi-Model Comparison

```bash
python run_eval.py \
    --input exports/trajectories \
    --model-paths \
        model_cache/qwen3_guard_8b \
        model_cache/qwen3_guard_4b \
        model_cache/llama3-guard-8B \
    --mode 3 \
    --output-dir results/comparison
```

#### Custom Prompt Styles

```bash
# Use SFT-aligned flat prompt (default)
python run_eval.py \
    --model-paths model_cache/qwen3_guard_8b \
    --prompt-style sft_flat \
    --chat-template plain \
    --mode 3

# Use original response moderation format
python run_eval.py \
    --model-paths model_cache/qwen3_guard_8b \
    --prompt-style response_moderation \
    --chat-template model \
    --mode 2
```

---

### Supervised Fine-Tuning

#### Prepare Training Data

```bash
python run_sft.py \
    --input exports/trajectories \
    --dataset data/subset.json \
    --mode 3 \
    --model-path model_cache/qwen3_guard_8b \
    --model-type qwen3 \
    --output-dir sft_runs/qwen3_exp1 \
    --annotation-path exports/trajectories/results.csv \
    --dry-run
```

This generates:
- `sft_runs/qwen3_exp1/data/braveguard_sft.json` - Training data in Alpaca format
- `sft_runs/qwen3_exp1/train.yaml` - LLaMA-Factory configuration
- `sft_runs/qwen3_exp1/export.yaml` - LoRA merge configuration

#### Run LoRA Fine-Tuning

```bash
python run_sft.py \
    --input exports/trajectories \
    --dataset data/subset.json \
    --mode 3 \
    --model-path model_cache/qwen3_guard_8b \
    --model-type qwen3 \
    --template qwen3 \
    --output-dir sft_runs/qwen3_exp1 \
    --epochs 5 \
    --learning-rate 2e-5 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --balance-labels oversample \
    --export-after-train
```

**Key Parameters**:
- `--balance-labels`: Handle class imbalance (`oversample`, `undersample`, or `none`)
- `--export-after-train`: Automatically merge LoRA weights into full model
- `--cutoff-len`: Maximum sequence length (default: 32768)

#### Evaluate Fine-Tuned Model

```bash
python run_eval.py \
    --input exports/test_set \
    --model-paths sft_runs/qwen3_exp1/merged \
    --model-type qwen3 \
    --prompt-style sft_flat \
    --mode 3 \
    --annotation-path exports/test_set/results.csv
```

---

## 🤖 Supported Models

| Model | Size | Type | Output Format |
|-------|------|------|---------------|
| **Qwen3Guard** | 4B/8B | Binary classifier | `safe` / `unsafe` |
| **LlamaGuard 3** | 8B | Policy-based | `safe` / `unsafe` |
| **NemoGuard** | 8B | Multi-category | `safe` / `unsafe` + categories |
| **XGuard** | 7B | Multilingual | `safe` / `unsafe` |
| **Custom Models** | Any | Extensible | Define your own parser |

### Adding Custom Models

1. **Create Engine**: Extend `BaseGuardEngine` in `evaluator/model_engine.py`
2. **Create Parser**: Extend `BaseResultParser` in `evaluator/result_parser.py`
3. **Register Factory**: Update `detect_model_type()` in `evaluator/factory.py`

Example:

```python
# In model_engine.py
class MyCustomGuardEngine(BaseGuardEngine):
    def get_model_name(self) -> str:
        return "mycustom"

    def build_messages(self, prompt: str) -> List[Dict]:
        return [{"role": "user", "content": prompt}]

# In factory.py
def detect_model_type(model_path: str) -> str:
    if "mycustom" in model_path.lower():
        return "mycustom"
    # ... existing logic
```

---

## 📊 Benchmarks

### Dataset Statistics

| Dataset | Trajectories | Categories | Jailbreak Methods | Avg Steps |
|---------|-------------|------------|-------------------|-----------|
| **BraveGuard-Full** | 2,000 | 8 | 12 | 6.3 |
| **ATBench-500** | 500 | 6 | 8 | 4.8 |
| **ASSE-Safety** | 1,200 | 10 | 15 | 7.1 |

### Baseline Results

**Guard Model Performance on BraveGuard-Full (Mode 3)**:

| Model | Size | Accuracy | Precision | Recall | F1 | ASR ↓ |
|-------|------|----------|-----------|--------|----|----|
| Qwen3Guard | 8B | 67.3% | 71.2% | 68.4% | 69.7% | 32.7% |
| Qwen3Guard | 4B | 61.8% | 65.9% | 62.1% | 64.0% | 38.2% |
| LlamaGuard 3 | 8B | 58.4% | 60.7% | 59.2% | 59.9% | 41.6% |
| **+ SFT (Ours)** | 8B | **82.6%** | **85.3%** | **83.1%** | **84.2%** | **17.4%** |

*ASR = Attack Success Rate (lower is better)*

---

## ⚙️ Configuration

### Command-Line Arguments

#### Evaluation (`run_eval.py`)

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `exports_v8` | Directory containing trajectory JSONL files |
| `--model-paths` | `-m` | Required | Space-separated list of guard model paths |
| `--output-dir` | - | `guard` | Output directory for CSV results |
| `--mode` | - | `3` | Evaluation mode (1/2/3) |
| `--dataset` | `-d` | `data/subset.json` | Dataset JSON (required for mode 1) |
| `--annotation-path` | - | Auto-detect | Ground truth labels for accuracy calculation |
| `--tp` | - | `8` | Tensor parallelism size |
| `--batch-size` | - | `32` | Inference batch size |
| `--max-model-len` | - | `32768` | Maximum context length |
| `--prompt-style` | - | `sft_flat` | Prompt format (`sft_flat` or `response_moderation`) |
| `--chat-template` | - | `auto` | Chat template source (`auto`, `plain`, or `model`) |

#### Fine-Tuning (`run_sft.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | Required | Trajectory directory |
| `--model-path` | Required | Base guard model path |
| `--model-type` | Auto-detect | Model type (`qwen3`, `llama3`) |
| `--output-dir` | Required | SFT output directory |
| `--epochs` | `3` | Number of training epochs |
| `--learning-rate` | `2e-5` | Learning rate |
| `--lora-rank` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA alpha parameter |
| `--balance-labels` | `none` | Label balancing strategy |
| `--export-after-train` | `False` | Auto-merge LoRA after training |
| `--dry-run` | `False` | Only generate configs, don't train |

### Configuration Files

**Main Config** (`config/config.json`):
```json
{
  "openai": {
    "url": "https://api.openai.com/v1",
    "keys": ["sk-..."],
    "model": "gpt-4-turbo"
  },
  "docker": {
    "image": "openclaw/agent:latest",
    "timeout": 300
  },
  "evaluation": {
    "max_workers": 16,
    "retry_limit": 3
  }
}
```

---

## 🔬 Advanced Usage

### Batch Processing with Multiple GPUs

```bash
# Distribute evaluation across multiple nodes
python run_eval.py \
    --input exports/large_dataset \
    --model-paths model_cache/qwen3_guard_8b \
    --tp 4 \
    --batch-size 64 \
    --max-model-len 16384
```

### Custom Evaluation Metrics

```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('results/qwen3_guard_8b_mode3.csv')

# Detailed classification report
print(classification_report(
    df['expected_harmful'],
    df['harmful'],
    target_names=['Safe', 'Unsafe']
))

# Confusion matrix
cm = confusion_matrix(df['expected_harmful'], df['harmful'])
print(f"Confusion Matrix:\n{cm}")
```

### Integration with R-Judge

```bash
# Use LLM-based judge for trajectory evaluation
python r-judge.py \
    --input exports/trajectories \
    --config config/llm_judge.yaml \
    --output exports/r_judge_results.json
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black evaluator/ run_*.py
isort evaluator/ run_*.py
```

### Adding New Features

- **New Guard Models**: Add to `evaluator/model_engine.py` and `factory.py`
- **New Evaluation Modes**: Extend `prompt_builder.py`
- **New Datasets**: Add loaders to `evaluator/data_loader.py`
- **New Metrics**: Extend `csv_writer.py` and `pipeline.py`

---

## 📚 Documentation

For more detailed documentation, please refer to:

- **[Paper](EMNLP_26.pdf)**: Full research paper (EMNLP 2026)
- **[API Documentation](docs/api.md)**: Detailed API reference
- **[Tutorial Notebooks](notebooks/)**: Jupyter notebooks with examples
- **[FAQ](docs/faq.md)**: Frequently asked questions

---

## 📝 Citation

If you find BraveGuard useful in your research, please consider citing:

```bibtex
@inproceedings{braveguard2026,
  title={BraveGuard: Evaluating Guard Models on AI Agent Trajectory Safety},
  author={Your Name et al.},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenClaw** team for agent execution infrastructure
- **LLaMA-Factory** for the fine-tuning framework
- **vLLM** for high-performance inference
- **Qwen** and **Meta** for open-sourcing guard models

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/BraveGuard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/BraveGuard/discussions)
- **Email**: your.email@example.com

---

<div align="center">

  **Built with ❤️ for AI Safety Research**

  ⭐ Star us on GitHub — it motivates us a lot!

</div>
