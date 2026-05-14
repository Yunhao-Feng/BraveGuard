"""OpenClaw 配置文件更新工具"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def update_openclaw_config(
    openclaw_config_path: str,
    agent_config: Dict[str, str]
) -> str:
    """
    更新 openclaw.json 配置文件中的 agent 模型设置

    Args:
        openclaw_config_path: openclaw.json 文件路径
        agent_config: agent 配置 (base_url, api_key, model)

    Returns:
        更新后的配置内容（JSON 字符串）
    """
    config_path = Path(openclaw_config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {openclaw_config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        openclaw_config = json.load(f)

    # 更新 provider 配置
    providers = openclaw_config.setdefault('models', {}).setdefault('providers', {})

    # 获取第一个 provider（通常是 Idealab）
    provider_name = list(providers.keys())[0] if providers else 'Idealab'
    provider_config = providers.setdefault(provider_name, {})

    # 更新 API 配置
    provider_config['baseUrl'] = agent_config['base_url']
    provider_config['apiKey'] = agent_config['api_key']

    # 更新模型配置
    models = provider_config.setdefault('models', [])
    if models:
        models[0]['id'] = agent_config['model']
        models[0]['name'] = agent_config['model']
    else:
        models.append({
            'id': agent_config['model'],
            'name': agent_config['model'],
            'api': 'openai-completions',
            'reasoning': False,
            'input': ['text'],
            'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
            'contextWindow': 400000,
            'maxTokens': 128000,
            'compat': {'maxTokensField': 'max_completion_tokens'}
        })

    # 更新 agents.defaults.model.primary
    agent_defaults = openclaw_config.setdefault('agents', {}).setdefault('defaults', {})
    agent_defaults.setdefault('model', {})['primary'] = f"{provider_name}/{agent_config['model']}"

    # 保存更新后的配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(openclaw_config, f, ensure_ascii=False, indent=2)

    logger.info(f"已更新配置文件: {openclaw_config_path}")

    return json.dumps(openclaw_config, ensure_ascii=False, indent=2)
