#!/usr/bin/env python3
"""
转换 data/atbench500.json 到 data/subset.json 格式
"""
import json
import time
import requests
import yaml
from pathlib import Path
from typing import List, Dict, Any

# 读取配置
with open('config/generate.yaml', 'r') as f:
    config = yaml.safe_load(f)

OPENAI_URL = config['openai']['url']
OPENAI_KEY = config['openai']['key']
MODEL_NAME = "gpt-5.4-mini-0317-global"

# 输入输出文件路径
INPUT_FILE = Path('data/atbench500.json')
OUTPUT_FILE = Path('data/atbench_trans.json')

# 创建输出目录
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def call_openai_api(prompt: str, max_retries: int = 5, retry_delay: float = 2.0) -> str:
    """调用 OpenAI API 生成 comment，带重试机制（固定延迟）"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OPENAI_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个安全专家，擅长分析 AI Agent 的越狱攻击。请用简洁的中文描述这个攻击的核心思路和特点。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"  API 返回错误状态码 {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    print(f"  等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return "API 调用失败，无法生成评论。"

        except requests.exceptions.Timeout:
            print(f"  API 调用超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                return "API 调用超时，无法生成评论。"

        except Exception as e:
            print(f"  API 调用异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                return f"API 调用失败: {str(e)}"

    return "API 调用失败，无法生成评论。"

def extract_user_query(content: List[List[Dict]]) -> str:
    """从 content 中提取 role 为 user 的 content"""
    for conv in content:
        for msg in conv:
            if msg.get('role') == 'user':
                return msg.get('content', '')
    return ''

def extract_agent_actions(content: List[List[Dict]]) -> List[str]:
    """从 content 中提取 agent 的 thought 和 action 拼接"""
    actions = []
    for conv in content:
        for msg in conv:
            if msg.get('role') == 'agent':
                thought = msg.get('thought', '').strip()
                action = msg.get('action', '').strip()

                # 拼接 thought 和 action
                if thought and action:
                    actions.append(f"{thought} {action}")
                elif action:
                    actions.append(action)
                elif thought:
                    actions.append(thought)

    return actions

def generate_comment(query: str, decomposed_query: List[str], category: str, jailbreak_method: str) -> str:
    """生成评论"""
    prompt = f"""请分析以下 AI Agent 越狱攻击案例：

类别：{category}
越狱方法：{jailbreak_method}
原始恶意查询：{query}
分解后的查询步骤：
{chr(10).join([f'{i+1}. {step}' for i, step in enumerate(decomposed_query)])}

请用 1-2 句话简洁描述这个攻击的核心思路和特点。"""

    return call_openai_api(prompt)

def convert_item(item: Dict[str, Any], item_id: int) -> Dict[str, Any]:
    """转换单个条目"""
    query = extract_user_query(item.get('content', []))
    decomposed_query = extract_agent_actions(item.get('content', []))
    category = item.get('real_world_harm', 'unknown')
    jailbreak_method = item.get('failure_mode', 'unknown')

    # 生成 comment
    print(f"正在处理条目 {item_id}，调用 API 生成评论...")
    comment = generate_comment(query, decomposed_query, category, jailbreak_method)

    return {
        "id": item_id,
        "category": category,
        "jailbreak_method": jailbreak_method,
        "query": query,
        "decomposed_query": decomposed_query,
        "comment": comment
    }

def main():
    """主函数"""
    print(f"正在读取输入文件: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总共需要处理 {len(data)} 个条目")

    # 如果输出文件已存在，读取已处理的数据
    processed_data = []
    if OUTPUT_FILE.exists():
        print(f"检测到已存在的输出文件，继续从断点处理...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        print(f"已处理 {len(processed_data)} 个条目，从第 {len(processed_data) + 1} 个继续")

    start_idx = len(processed_data)

    # 转换数据
    for idx in range(start_idx, len(data)):
        item = data[idx]
        item_id = idx + 1

        print(f"\n处理条目 {item_id}/{len(data)}")
        converted_item = convert_item(item, item_id)
        processed_data.append(converted_item)

        # 实时落盘保存
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 条目 {item_id} 已完成并保存")

        # 避免请求过快，稍微延迟
        if idx < len(data) - 1:  # 不是最后一个
            time.sleep(0.5)

    print(f"\n全部完成！共转换 {len(processed_data)} 个条目")
    print(f"输出文件: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
