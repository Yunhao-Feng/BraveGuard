<div align="center">
  <img src="assets/logo.png" alt="BraveGuard logo" width="180" />

BraveGuard is a research framework for **trajectory-level safety evaluation and guard model training for computer-use agents**.

Unlike conventional moderation systems that operate on isolated prompts or responses, BraveGuard focuses on the more realistic and challenging setting of **multi-step agent trajectories**, where risks emerge through sequences of actions, tool interactions, intermediate reasoning, and accumulated side effects.

---

## ✨ Key Features

BraveGuard provides a complete pipeline for building and evaluating trajectory-aware guard models:

- **Open-world threat mining** from emerging security and safety sources.
- **Executable task synthesis** that converts threat knowledge into realistic agent tasks.
- **Trajectory generation pipeline** with real agent execution and attack pressure.
- **Unified evaluation framework** supporting multiple guard model families.
- **Trajectory-aware SFT data construction** for safety alignment.
- **Self-evolving defense loop** that continuously discovers new threats, collects failures, and improves guard capabilities.

---

## Overview

BraveGuard transforms open-world threat intelligence into trajectory-level supervision for computer-use agents.

The framework continuously:

1. Mines emerging threats from public sources.
2. Converts threats into executable attack tasks.
3. Collects agent trajectories using OpenClaw.
4. Annotates trajectories with safety labels and rationales.
5. Trains trajectory-aware guard models.
6. Feeds hard cases back into the next training cycle.

This creates a **self-evolving defense loop** that adapts to newly emerging risks over time.

<p align="center">
  <img src="assets/braveguard_method.png" alt="Overview of the BraveGuard self-evolving defense loop" width="900" />
</p>

---

## 📊 Main Results

BraveGuard significantly improves trajectory-level safety detection on challenging computer-use agent benchmarks.

### AgentHazard-Strongest (GPT-5.5 + OpenClaw)

| Model                     |        Acc. (%) | Rec. (%) |          F1 (%) |
| ------------------------- | --------------: | -------: | --------------: |
| AgentDoG-Llama3.1-8B      |           64.26 |    58.97 |           70.99 |
| AgentDoG-Qwen2.5-7B       |           65.02 |    60.51 |           71.95 |
| BraveGuard-Llama-Guard-8B |           82.51 |    92.82 |           88.73 |
| BraveGuard-Qwen3-Guard-8B | **83.65** |    91.28 | **89.22** |
| BraveGuard-Qwen3-Guard-4B |           80.99 |    88.72 |           87.37 |

### ATBench-500

| Model                     |        Acc. (%) |        Rec. (%) |          F1 (%) |
| ------------------------- | --------------: | --------------: | --------------: |
| AgentDoG-Qwen2.5-7B       | **87.40** |           95.60 |           88.40 |
| AgentDoG-Llama3.1-8B      |           87.60 | **98.40** | **88.80** |
| BraveGuard-Qwen3-Guard-8B |           86.40 |           95.20 |           86.10 |

### Highlights

- Average off-the-shelf guard accuracy on AgentHazard improves from **38.79% → 82.38%**.
- BraveGuard synthesizes **7,308 executable tasks**.
- Coverage includes **28 risk categories** and **32 attack methods**.
- Each task contains an average of **3.36 decomposed execution steps**.

---

## 📈 Category-wise Performance

BraveGuard demonstrates strong performance across most AgentHazard categories while maintaining competitive generalization on ATBench.

Remaining challenges such as **data exfiltration**, **compliance bypass**, and other advanced attack scenarios highlight promising directions for future work.

<p align="center">
  <img src="assets/category_performance.png" alt="Category-wise BraveGuard performance on AgentHazard-Strongest and ATBench-500" width="900" />
</p>

---

## 🏗 Repository Structure

```text
.
├── generate.py
├── run_eval.py
├── evaluator/
├── sft/
├── data/
├── rock_runner.py
└── local_runner.py
```

### Components

| Module              | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| `generate.py`     | Generate or replay agent trajectories                                           |
| `run_eval.py`     | Batch evaluation entrypoint                                                     |
| `evaluator/`      | Prompt construction, model adapters, parsing, metrics, and evaluation pipelines |
| `rock_runner.py`  | ROCK execution backend                                                          |
| `local_runner.py` | Local execution backend                                                         |
| `sft/`            | Supervised fine-tuning data construction                                        |
| `data/`           | Public benchmark and task resources                                             |

---

## 🔬 Evaluation Modes

BraveGuard supports three evaluation settings with different observability assumptions:

| Mode   | Input                                            |
| ------ | ------------------------------------------------ |
| Mode 1 | Trajectory + attack metadata                     |
| Mode 2 | Trajectory + safety policy / evaluation criteria |
| Mode 3 | Pure trajectory-only judgment                    |

These modes enable controlled studies of guard robustness under varying levels of available context.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate braveguard
```

### 2. Configure Credentials

Fill in the following configuration files with your own credentials and endpoints:

```text
config/config.json
config/llm_judge.yaml
config/openclaw.json
```

### 3. Generate Agent Trajectories

```bash
python generate.py
```

### 4. Evaluate Guard Models

```bash
python run_eval.py \
  --input tmp/workspace/data/agenthazard_strongest \
  --model-paths /path/to/guard-model \
  --mode 3 \
  --output-dir results
```

---

## 🔒 Security Notes

Before running experiments:

- Never commit real API keys or access tokens.
- Store secrets in environment variables or untracked local files.
- Treat collected trajectories as potentially sensitive data.
- Sanitize logs and trajectories before public release.


---

## License

Released under the MIT License.
