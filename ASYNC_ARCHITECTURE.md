# vLLM 评估流程修复总结

## 修复内容

### 问题 1：超长 prompt 导致崩溃 ✅ 已修复

**原因：** 数据集中存在超长 prompt（如 151290 tokens），超过模型 `max_model_len=32768`

**解决方案：**
- 在 `model_engine.py` 的 `batch_generate()` 方法中添加了完整的 token 长度检查
- 超长 prompt 自动截断到 32000 tokens（给输出留 768 tokens 余量）
- 截断策略：保留最后 N 个 tokens（保留最新上下文）
- 详细日志记录哪些样本被截断

**关键代码：** `evaluator/model_engine.py:99-150`

```python
def batch_generate(
    self,
    prompts: List[str],
    session_ids: List[int] = None,
    max_input_tokens: int = 32000,
) -> List[str]:
    # tokenize 并检查长度
    for idx, (prompt, session_id) in enumerate(zip(prompts, session_ids)):
        input_ids = self.tokenizer.encode(prompt)
        original_len = len(input_ids)

        if original_len > max_input_tokens:
            logger.warning(f"Session {session_id}: 截断 {original_len} -> {max_input_tokens} tokens")
            input_ids = input_ids[-max_input_tokens:]
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        processed_prompts.append(prompt)

    # 批量推理
    outputs = self.llm.generate(processed_prompts, self.sampling_params)
    return results
```

---

### 问题 2：异步并发调用导致卡住 ✅ 已修复

**原因：** 多个 async task 并发调用同一个 `self.llm.generate()`，vLLM offline API 不支持这种共享实例的高并发调用

**解决方案：**
- 完全移除异步并发模式（去掉 `asyncio`）
- 改为 batch 串行推理：按批次调用 `engine.batch_generate()`
- 保持单条样本异常不影响整体流程
- 结果顺序与输入顺序一致

**关键代码：** `evaluator/pipeline.py:47-140`

```python
def run(self):
    # Step 4: batch 串行推理
    batch_size = self.config.batch_size
    for batch_start in range(0, self._total_count, batch_size):
        batch_end = min(batch_start + batch_size, self._total_count)
        batch_data = tasks_data[batch_start:batch_end]

        # 准备 batch
        batch_session_ids = [...]
        batch_prompts = [...]

        # batch 推理（串行调用 vLLM）
        raw_outputs = self.engine.batch_generate(
            batch_prompts,
            session_ids=batch_session_ids,
            max_input_tokens=32000,
        )

        # 解析结果并写入 CSV
        for session_id, raw_output in zip(batch_session_ids, raw_outputs):
            result = self.parser.parse(session_id, raw_output)
            self.csv_writer.append(result)
```

---

## 修改文件清单

### 1. `evaluator/model_engine.py`
- ✅ 添加 `batch_generate()` 方法
- ✅ 集成 token 长度检查和截断逻辑
- ✅ 添加详细日志记录

### 2. `evaluator/pipeline.py`
- ✅ 移除 `asyncio` 相关代码
- ✅ 删除 `_process_single_item()` 和 `_run_async()` 方法
- ✅ 重写 `run()` 方法为 batch 串行推理
- ✅ 保持异常处理和进度输出

### 3. `evaluator/config.py`
- ✅ 添加 `batch_size: int = 32` 参数
- ✅ 标记 `max_concurrent` 为已废弃（保留兼容性）

### 4. `run_eval.py`
- ✅ 添加 `--batch-size` 命令行参数（默认 32）
- ✅ 移除 `asyncio` 导入
- ✅ 更新配置创建和打印信息

---

## 使用方式

### 基本用法（与之前相同）

```bash
python run_eval.py \
  --input ./exports_v8 \
  --model-path qwen3_guard_8b \
  --output ./guard/output_mode1.csv \
  --mode 1 \
  --batch-size 32
```

### 新参数说明

- `--batch-size`：batch 推理的批次大小（默认 32）
  - 建议值：16-64，根据 GPU 显存调整
  - 越大吞吐量越高，但显存占用也越大

### 已废弃参数

- `--max-concurrent`：已不再使用（保留兼容性，不影响运行）

---

## 关键改进

### ✅ 稳定性
- 不再出现 `process_input_sockets` 线程异常
- 不再因超长 prompt 崩溃
- 单条样本异常不影响整体流程

### ✅ 可控性
- 超长样本自动截断到 32000 tokens
- 详细日志记录截断信息
- 结果顺序与输入顺序一致

### ✅ 性能
- batch 推理充分利用 vLLM 的批处理能力
- 避免了异步并发的调度开销
- 可通过 `--batch-size` 调整吞吐量

---

## 工作流程

```
准备阶段：
1. 加载所有轨迹文件
2. 构建所有任务（session_id + messages）
3. 初始化 vLLM 引擎和 CSV 文件

执行阶段（batch 串行推理）：
┌─────────────────────────────────────────────┐
│  Batch 1: [Item1, Item2, ..., Item32]      │
│     ↓                                        │
│  构建 prompts (检查长度、截断)               │
│     ↓                                        │
│  vLLM.generate(batch_prompts)  [一次调用]   │
│     ↓                                        │
│  解析结果 + 写入 CSV + 输出准确率            │
├─────────────────────────────────────────────┤
│  Batch 2: [Item33, Item34, ..., Item64]    │
│     ↓                                        │
│  ...                                         │
└─────────────────────────────────────────────┘
```

---

## 验证要点

### 运行前检查
```bash
# 1. 检查模型路径
ls -lh qwen3_guard_8b/

# 2. 检查输入数据
ls -lh exports_v8/ | head

# 3. 检查 GPU 可用性
nvidia-smi
```

### 运行时监控
```bash
# 实时查看日志（观察是否有截断警告）
tail -f <日志文件>

# 观察 GPU 显存占用
watch -n 1 nvidia-smi
```

### 运行后验证
- ✅ 无 `ValueError: The decoder prompt ... is longer than` 错误
- ✅ 无 `Exception in thread Thread-1 (process_input_sockets)` 错误
- ✅ 评估流程正常完成
- ✅ 输出 CSV 包含所有样本
- ✅ 日志中可见超长样本截断信息

---

## 注意事项

1. **不要强行调大 `max_model_len`**：模型原生最大长度就是 32768
2. **不要再使用异步并发模式**：vLLM offline API 不支持共享实例并发
3. **根据 GPU 显存调整 `batch_size`**：
   - 8xA100 80GB：可用 32-64
   - 8xV100 32GB：建议 16-32
   - 单卡：建议 4-8

---

## 修复时间

**修复日期：** 2026-05-12
**修复版本：** v2.0 (batch 串行推理)
