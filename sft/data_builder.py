"""Build LLaMA-Factory SFT datasets from OpenClaw trajectory exports."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from evaluator.annotations import load_annotations, resolve_annotation_path, row_to_label
from evaluator.data_loader import DatasetLoader, TrajectoryLoader
from evaluator.prompt_builder import PromptBuilder


@dataclass(frozen=True)
class SFTExample:
    """One Alpaca-style SFT record for LLaMA-Factory."""

    instruction: str
    input: str
    output: str
    system: str
    session_id: int

    def to_llamafactory_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "system": self.system,
            "session_id": self.session_id,
        }


class GuardSFTDataBuilder:
    """Convert trajectory JSONL files into labeled guard SFT examples."""

    DEFAULT_SYSTEM_PROMPT = PromptBuilder.DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        input_dir: str,
        dataset_path: str,
        mode: int,
        model_type: str,
        fallback_label: Optional[str] = None,
        category: str = "Jailbreak",
        refusal: str = "No",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        annotation_path: Optional[str] = None,
        seed: int = 42,
    ):
        if mode not in {1, 2, 3}:
            raise ValueError("mode 必须是 1、2 或 3")
        if model_type not in {"qwen3", "llama3"}:
            raise ValueError("model_type 必须是 qwen3 或 llama3")
        if fallback_label is not None and fallback_label.lower() not in {"safe", "unsafe"}:
            raise ValueError("fallback_label 必须是 safe 或 unsafe")

        self.input_dir = Path(input_dir)
        self.dataset_path = dataset_path
        self.mode = mode
        self.model_type = model_type
        self.fallback_label = fallback_label.lower() if fallback_label else None
        self.category = category
        self.refusal = refusal
        self.system_prompt = system_prompt
        self.annotation_path = self._resolve_annotation_path(annotation_path)
        self.seed = seed
        self.trajectory_loader = TrajectoryLoader(input_dir)
        self.dataset_loader = DatasetLoader(dataset_path) if mode == 1 else None
        self.prompt_builder = PromptBuilder(prompt_style="sft_flat", system_prompt=system_prompt)
        self.annotations = load_annotations(self.annotation_path)

    def build_examples(self) -> list[SFTExample]:
        """Build all available SFT examples sorted by session id."""
        dataset_map = self.dataset_loader.load() if self.dataset_loader else {}
        examples: list[SFTExample] = []

        for filepath in self.trajectory_loader.list_session_files():
            session_id = self.trajectory_loader.extract_session_id(filepath)
            trajectory = self.trajectory_loader.load_trajectory(filepath)
            if not trajectory:
                continue

            metadata = dataset_map.get(session_id) if self.mode == 1 else None
            if self.mode == 1 and metadata is None:
                continue

            instruction, trajectory_text = self.prompt_builder.build_instruction_and_trajectory(
                mode=self.mode,
                trajectory=trajectory,
                metadata=metadata,
            )
            flat_prompt = self.prompt_builder.build_flat_prompt(instruction, trajectory_text)
            output = self._format_output(session_id)
            examples.append(
                SFTExample(
                    instruction=flat_prompt,
                    input="",
                    output=output,
                    system=self.system_prompt,
                    session_id=session_id,
                )
            )

        return examples

    def write_dataset(
        self,
        output_dir: str,
        dataset_name: str,
        val_size: float = 0.0,
    ) -> Dict[str, Path]:
        """Write train/eval JSON files and LLaMA-Factory dataset_info.json."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        examples = self.build_examples()
        if not examples:
            raise RuntimeError("没有可用于 SFT 的轨迹样本")

        train_examples, eval_examples = self._split_examples(examples, val_size)
        train_file = output_path / f"{dataset_name}.json"
        eval_file = output_path / f"{dataset_name}_eval.json"
        info_file = output_path / "dataset_info.json"

        self._write_json(train_file, [e.to_llamafactory_dict() for e in train_examples])
        if eval_examples:
            self._write_json(eval_file, [e.to_llamafactory_dict() for e in eval_examples])

        dataset_info = {
            dataset_name: {
                "file_name": train_file.name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                },
            }
        }
        if eval_examples:
            dataset_info[f"{dataset_name}_eval"] = {
                "file_name": eval_file.name,
                "columns": dataset_info[dataset_name]["columns"],
            }
        self._write_json(info_file, dataset_info)

        return {
            "dataset_dir": output_path,
            "train_file": train_file,
            "eval_file": eval_file if eval_examples else Path(),
            "dataset_info": info_file,
        }

    def _format_output(self, session_id: int) -> str:
        row = self.annotations.get(session_id)
        if row is None:
            if self.fallback_label is None:
                raise KeyError(
                    f"Session {session_id} 在标注文件 {self.annotation_path} 中没有 harmful/label 记录；"
                    "请补齐 CSV/JSON，或显式传 --fallback-label safe/unsafe。"
                )
            row = {"label": self.fallback_label}

        label = row_to_label(row)
        category = str(row.get("category") or row.get("categories") or self.category)
        refusal = str(row.get("refusal") or self.refusal)

        if self.model_type == "llama3":
            return "unsafe" if label == "unsafe" else "safe"

        safety = "Unsafe" if label == "unsafe" else "Safe"
        category = category if label == "unsafe" else "None"
        return f"Safety: {safety}\nCategories: {category}\nRefusal: {refusal}"

    def _split_examples(
        self,
        examples: list[SFTExample],
        val_size: float,
    ) -> tuple[list[SFTExample], list[SFTExample]]:
        if val_size <= 0:
            return examples, []
        if val_size >= 1:
            raise ValueError("val_size 必须小于 1")

        shuffled = examples[:]
        random.Random(self.seed).shuffle(shuffled)
        eval_count = max(1, int(len(shuffled) * val_size))
        eval_set = sorted(shuffled[:eval_count], key=lambda x: x.session_id)
        train_set = sorted(shuffled[eval_count:], key=lambda x: x.session_id)
        return train_set, eval_set

    def _resolve_annotation_path(self, annotation_path: Optional[str]) -> Path:
        return resolve_annotation_path(self.input_dir, annotation_path)

    @staticmethod
    def _write_json(path: Path, data: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
