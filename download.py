#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/Llama-Guard-3-8B', local_dir="model_cache/llama3-guard-8B")