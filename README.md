# BraveGuard

BraveGuard is a research framework for **trajectory-level safety evaluation** of LLM agents. It focuses on a harder setting than single-turn moderation: deciding whether a full multi-step agent trajectory is safe or unsafe, including tool calls, intermediate reasoning traces, and side effects.

> Note: this repository corresponds to a paper submission and does **not** claim acceptance at EMNLP.

## Why BraveGuard

Most guard models are trained and evaluated on user prompts or model responses. In real agent deployments, risk emerges across a sequence of actions. BraveGuard provides:

- **Trajectory generation pipeline** with realistic task execution and attack pressure.
- **Unified guard evaluation engine** for multiple guard families.
- **Three evaluation modes** that vary how much metadata is available to the guard.
- **SFT data construction utilities** to improve trajectory-aware guard behavior.

## Core Components

- `generate.py`: Generates or replays agent trajectories.
- `run_eval.py`: Main entry for batch guard-model evaluation.
- `evaluator/`: Prompt building, model adapters, parsing, metrics, and pipeline logic.
- `rock_runner.py` / `local_runner.py`: Runtime backends for trajectory execution.
- `sft/`: Data construction for supervised fine-tuning.
- `data/`: Public benchmark/task files used by the project.

## Evaluation Modes

BraveGuard supports three prompt modes for controlled ablations:

1. **Mode 1**: trajectory + attack metadata.
2. **Mode 2**: trajectory + policy/evaluation criteria.
3. **Mode 3**: pure trajectory judgment with minimal hints.

This design allows studying guard robustness under both ideal and realistic observability.

## Quick Start

### 1) Environment

```bash
conda env create -f environment.yml
conda activate braveguard
```

### 2) Configure keys and endpoints

Copy template configs and fill in your own credentials:

- `config/config.json`
- `config/llm_judge.yaml`
- `config/openclaw.json`

### 3) Generate trajectories

```bash
python generate.py
```

### 4) Evaluate guard models

```bash
python run_eval.py \
  --input tmp/workspace/data/agenthazard_strongest \
  --model-paths /path/to/guard-model \
  --mode 3 \
  --output-dir results
```

## Repository Hygiene & Security

This repo intentionally uses **placeholder credentials** in tracked configs. Before running experiments:

- Never commit real API keys/tokens.
- Keep secrets in local environment variables or untracked files.
- Treat exported trajectories as potentially sensitive and sanitize before sharing.

## Citation

If you find this project useful, please cite the associated paper in `EMNLP_26.pdf`.

## License

MIT License.
