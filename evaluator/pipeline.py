"""评估流水线：串联所有模块"""

import logging
import time
from typing import List, Dict

from .config import EvalConfig
from .data_loader import TrajectoryLoader, DatasetLoader
from .prompt_builder import PromptBuilder
from .factory import create_engine_and_parser
from .base_parser import GuardResult
from .csv_writer import CSVWriter
from .metrics import BinaryMetrics, load_reference_labels

logger = logging.getLogger(__name__)

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class EvalPipeline:
    """
    评估流水线（batch 串行版本）。

    流程：
    1. 加载轨迹文件和数据集元信息
    2. 根据模式构建 prompt
    3. 按 batch 串行调用 vLLM 推理
    4. 解析结果并写入 CSV
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.trajectory_loader = TrajectoryLoader(config.input_dir)
        self.dataset_loader = DatasetLoader(config.dataset_path) if config.mode == 1 else None
        self.prompt_builder = PromptBuilder(prompt_style=config.prompt_style)
        # 使用工厂函数自动创建引擎和解析器
        self.engine, self.parser = create_engine_and_parser(config)
        self.csv_writer = CSVWriter(config.output_path)
        self._completed_count = 0
        self._harmful_count = 0
        self._total_count = 0
        self.reference_labels = load_reference_labels(config.input_dir, config.annotation_path)
        self.metrics = BinaryMetrics()

    def run(self):
        """执行完整评估流程（batch 串行推理）"""
        start_time = time.time()

        # Step 1: 加载轨迹文件
        print("[Step 1/4] 加载轨迹文件...")
        session_files = self.trajectory_loader.list_session_files()
        if not session_files:
            print("未找到任何轨迹文件，退出。")
            return

        # Step 2: 构建待处理任务
        print("[Step 2/4] 构建评估任务...")
        tasks_data = []  # [(session_id, messages), ...]
        skipped = 0

        # 如果是 mode1，预加载数据集
        dataset_map = None
        if self.config.mode == 1:
            dataset_map = self.dataset_loader.load()

        for filepath in session_files:
            session_id = self.trajectory_loader.extract_session_id(filepath)
            trajectory = self.trajectory_loader.load_trajectory(filepath)

            if not trajectory:
                logger.warning(f"Session {session_id}: 轨迹为空，跳过")
                skipped += 1
                continue

            # 获取元信息（mode1）
            metadata = None
            if self.config.mode == 1:
                metadata = dataset_map.get(session_id)
                if metadata is None:
                    logger.warning(
                        f"Session {session_id}: 数据集中无对应记录，跳过"
                    )
                    skipped += 1
                    continue

            # 构建消息
            messages = self.prompt_builder.build_messages(
                mode=self.config.mode,
                trajectory=trajectory,
                metadata=metadata,
            )

            tasks_data.append((session_id, messages))

        self._total_count = len(tasks_data)
        print(f"  待评估: {self._total_count} 条, 跳过: {skipped} 条")
        if self.reference_labels:
            matched = sum(1 for session_id, _ in tasks_data if session_id in self.reference_labels)
            print(f"  参考标注: {matched}/{self._total_count} 条可用于 accuracy/precision/recall/F1")

        if self._total_count == 0:
            print("无待评估数据，退出。")
            return

        # Step 3: 初始化模型和 CSV
        print("[Step 3/4] 初始化 vLLM 推理引擎...")
        self.engine.initialize()
        self.csv_writer.initialize()

        # Step 4: batch 串行推理
        print(f"[Step 4/4] Batch 串行推理（batch_size: {self.config.batch_size}）...")

        # 按 batch 处理
        batch_size = self.config.batch_size
        for batch_start in range(0, self._total_count, batch_size):
            batch_end = min(batch_start + batch_size, self._total_count)
            batch_data = tasks_data[batch_start:batch_end]

            # 准备 batch
            batch_session_ids = [session_id for session_id, _ in batch_data]
            batch_messages = [messages for _, messages in batch_data]

            # 构建 prompts
            batch_prompts = []
            for messages in batch_messages:
                try:
                    prompt = self.engine.build_prompt(messages)
                    batch_prompts.append(prompt)
                except Exception as e:
                    logger.error(f"构建 prompt 失败: {e}")
                    batch_prompts.append("")  # 空 prompt，后续会处理为错误

            # batch 推理
            try:
                raw_outputs = self.engine.batch_generate(
                    batch_prompts,
                    session_ids=batch_session_ids,
                    max_input_tokens=32000,
                )
            except Exception as e:
                logger.error(f"Batch 推理失败: {e}")
                # 如果整个 batch 失败，给所有样本标记错误
                raw_outputs = [""] * len(batch_prompts)

            # 解析结果并写入 CSV
            for session_id, raw_output in zip(batch_session_ids, raw_outputs):
                try:
                    if raw_output:
                        result = self.parser.parse(session_id, raw_output)
                    else:
                        # 空输出，标记为错误
                        result = GuardResult(
                            session_id=session_id,
                            safety_label=None,
                            categories=None,
                            refusal=None,
                            raw_output="",
                            harmful=False,
                            error="Empty output or failed to generate",
                        )

                    # 写入 CSV
                    self.csv_writer.append(result)

                    # 更新计数
                    self._completed_count += 1
                    if result.harmful:
                        self._harmful_count += 1

                    expected = self.reference_labels.get(session_id)
                    metric_text = f"harmful率: {self._harmful_count / self._completed_count:.4f} ({self._harmful_count}/{self._completed_count})"
                    if expected is not None:
                        self.metrics.update(expected, result.harmful)
                        metric_text = (
                            f"accuracy: {self.metrics.accuracy:.4f} "
                            f"({self.metrics.tp + self.metrics.tn}/{self.metrics.count}), "
                            f"P/R/F1: {self.metrics.precision:.4f}/{self.metrics.recall:.4f}/{self.metrics.f1:.4f}"
                        )

                    # 实时输出进度
                    status = "HARMFUL" if result.harmful else ("ERROR" if result.error else "SAFE")
                    print(f"[{self._completed_count}/{self._total_count}] "
                          f"Session {session_id}: {status} | "
                          f"{metric_text}")

                except Exception as e:
                    logger.error(f"Session {session_id} 解析失败: {e}")
                    # 即使失败也记录到 CSV
                    result = GuardResult(
                        session_id=session_id,
                        safety_label=None,
                        categories=None,
                        refusal=None,
                        raw_output="",
                        harmful=False,
                        error=str(e),
                    )
                    self.csv_writer.append(result)

                    # 更新计数
                    self._completed_count += 1
                    expected = self.reference_labels.get(session_id)
                    if expected is not None:
                        self.metrics.update(expected, False)
                        metric_text = (
                            f"accuracy: {self.metrics.accuracy:.4f} "
                            f"({self.metrics.tp + self.metrics.tn}/{self.metrics.count}), "
                            f"P/R/F1: {self.metrics.precision:.4f}/{self.metrics.recall:.4f}/{self.metrics.f1:.4f}"
                        )
                    else:
                        metric_text = f"harmful率: {self._harmful_count / self._completed_count if self._completed_count > 0 else 0.0:.4f} ({self._harmful_count}/{self._completed_count})"
                    print(f"[{self._completed_count}/{self._total_count}] "
                          f"Session {session_id}: ERROR | "
                          f"{metric_text}")

        elapsed = time.time() - start_time
        harmful_rate = self._harmful_count / self._total_count if self._total_count > 0 else 0.0

        print(f"\n[完成] 耗时: {elapsed:.1f}s, "
              f"平均: {elapsed / self._total_count:.2f}s/条")
        print(f"最终 harmful 预测率: {harmful_rate:.4f} ({self._harmful_count}/{self._total_count})")
        if self.metrics.count:
            print(
                "最终评估指标: "
                f"accuracy={self.metrics.accuracy:.4f}, "
                f"precision={self.metrics.precision:.4f}, "
                f"recall={self.metrics.recall:.4f}, "
                f"f1={self.metrics.f1:.4f}, "
                f"tp={self.metrics.tp}, tn={self.metrics.tn}, fp={self.metrics.fp}, fn={self.metrics.fn}"
            )

        if self.config.output_path:
            print(f"结果已写入: {self.config.output_path}")
        else:
            print("未配置 CSV 输出路径，结果未保存")
