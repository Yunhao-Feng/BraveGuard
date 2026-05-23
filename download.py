#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Shanghai_AI_Laboratory/AgentDoG-Qwen2.5-7B', local_dir="model_cache/AgentDoG-Qwen2.5-7B")