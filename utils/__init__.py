"""工具模块"""
from .retry import async_retry, safe_execute
from .judge import LLMJudge
from .config_updater import update_openclaw_config
from .key_pool import KeyPool
from .config_generator import ConfigGenerator

import yaml
import argparse
import ast
import json
import re
from typing import Any

def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def load_config_as_namespace(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)





def extract_json_list(text: str) -> list[dict]:
    """
    从一段混合文本中提取 JSON / Python-literal 风格的 list[dict]。

    支持场景：
    1. 整段文本本身就是 JSON 数组
    2. markdown code block: ```json ... ``` / ```python ... ``` / ```bash ... ```
    3. bash heredoc: cat << 'EOF' ... EOF
    4. 普通文本中夹带一个顶层 JSON 数组
    5. Python 字面量形式的 list[dict]

    返回：
        list[dict]

    异常：
        ValueError: 没找到合法的 list[dict]
    """

    def normalize_result(obj: Any) -> list[dict]:
        if not isinstance(obj, list):
            raise ValueError("parsed object is not a list")
        if not all(isinstance(x, dict) for x in obj):
            raise ValueError("parsed list is not list[dict]")
        return obj

    def try_json_loads(s: str) -> list[dict] | None:
        s = s.strip()
        if not s:
            return None
        try:
            return normalize_result(json.loads(s))
        except Exception:
            return None

    def try_python_literal(s: str) -> list[dict] | None:
        s = s.strip()
        if not s:
            return None
        try:
            obj = ast.literal_eval(s)
            return normalize_result(obj)
        except Exception:
            return None

    def try_parse_candidate(s: str) -> list[dict] | None:
        # 先按标准 JSON
        obj = try_json_loads(s)
        if obj is not None:
            return obj
        # 再按 Python 字面量
        obj = try_python_literal(s)
        if obj is not None:
            return obj
        return None

    def extract_code_blocks(s: str) -> list[str]:
        # 匹配 ```lang ... ``` 和 ``` ... ```
        pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)
        return [m.strip() for m in pattern.findall(s)]

    def extract_heredoc_blocks(s: str) -> list[str]:
        # 匹配:
        # cat << 'EOF'
        # ...
        # EOF
        pattern = re.compile(
            r"<<\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?.*?\n(.*?)\n\1",
            re.DOTALL,
        )
        return [content.strip() for _, content in pattern.findall(s)]

    def find_balanced_top_level_arrays(s: str) -> list[str]:
        """
        扫描文本中所有可能的顶层 [...] 片段。
        会正确跳过字符串里的括号。
        """
        results = []
        n = len(s)

        for start in range(n):
            if s[start] != "[":
                continue

            stack = ["["]
            in_string = False
            string_quote = ""
            escape = False

            for i in range(start + 1, n):
                ch = s[i]

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == string_quote:
                        in_string = False
                    continue

                if ch in ("'", '"'):
                    in_string = True
                    string_quote = ch
                elif ch == "[":
                    stack.append("[")
                elif ch == "]":
                    if not stack or stack[-1] != "[":
                        break
                    stack.pop()
                    if not stack:
                        results.append(s[start:i + 1])
                        break

        return results

    candidates: list[str] = []

    # 1. 整段直接尝试
    candidates.append(text.strip())

    # 2. code block 内容
    code_blocks = extract_code_blocks(text)
    candidates.extend(code_blocks)

    # 3. code block 里的 heredoc
    for block in code_blocks:
        candidates.extend(extract_heredoc_blocks(block))

    # 4. 全文 heredoc
    candidates.extend(extract_heredoc_blocks(text))

    # 5. 全文中扫描顶层 [...]
    candidates.extend(find_balanced_top_level_arrays(text))

    # 去重，保序
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = c.strip()
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(key)

    # 优先尝试更像结果的候选：以 [ 开头的优先，长度大的优先
    unique_candidates.sort(key=lambda x: (not x.lstrip().startswith("["), -len(x)))

    for candidate in unique_candidates:
        parsed = try_parse_candidate(candidate)
        if parsed is not None:
            return parsed

    raise ValueError("No valid list[dict] found in input text.")


__all__ = [
    'async_retry',
    'safe_execute',
    'LLMJudge',
    'update_openclaw_config',
    'KeyPool',
    'ConfigGenerator',
]
