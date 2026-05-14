"""动态配置文件生成器"""
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """
    基于模板动态生成 OpenClaw 配置文件
    每个任务使用独立的临时配置，避免并发覆盖
    """

    def __init__(self, template_path: str):
        """
        Args:
            template_path: openclaw.json 模板文件路径
        """
        self.template_path = Path(template_path)

        if not self.template_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {template_path}")

        with open(self.template_path, 'r', encoding='utf-8') as f:
            self.template = json.load(f)

        logger.info(f"配置模板加载: {template_path}")

    def generate_config(
        self,
        api_key: str,
        base_url: str,
        model: str,
        output_path: str
    ) -> str:
        """
        生成配置文件

        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
            output_path: 输出文件路径

        Returns:
            输出文件的绝对路径
        """
        config = json.loads(json.dumps(self.template))  # Deep copy

        # 更新 provider 配置
        providers = config.setdefault('models', {}).setdefault('providers', {})
        provider_name = list(providers.keys())[0] if providers else 'Idealab'
        provider_config = providers.setdefault(provider_name, {})

        provider_config['baseUrl'] = base_url
        provider_config['apiKey'] = api_key

        # 更新模型配置
        models = provider_config.setdefault('models', [])
        if models:
            models[0]['id'] = model
            models[0]['name'] = model
        else:
            models.append({
                'id': model,
                'name': model,
                'api': 'openai-completions',
                'reasoning': False,
                'input': ['text'],
                'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
                'contextWindow': 400000,
                'maxTokens': 128000,
                'compat': {'maxTokensField': 'max_completion_tokens'}
            })

        # 更新 agents.defaults.model.primary
        agent_defaults = config.setdefault('agents', {}).setdefault('defaults', {})
        agent_defaults.setdefault('model', {})['primary'] = f"{provider_name}/{model}"

        # 写入文件
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.debug(f"配置文件生成: {output_path}")
        return str(output.absolute())
