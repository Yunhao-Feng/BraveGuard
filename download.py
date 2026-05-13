#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3Guard-Gen-8B', local_dir="guard_model_cache")