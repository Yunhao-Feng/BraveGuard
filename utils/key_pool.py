"""API Key 轮询池"""
import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)


class KeyPool:
    """
    线程安全的 API Key 轮询池
    使用 round-robin 策略分配 key
    """

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("API keys 列表不能为空")

        self.keys = keys
        self.index = 0
        self.lock = asyncio.Lock()
        logger.info(f"KeyPool 初始化: {len(keys)} 个 API keys")

    async def get_key(self) -> str:
        """获取下一个 key（轮询）"""
        async with self.lock:
            key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)
            # 只记录 key 的前后各 4 个字符
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
            logger.debug(f"分配 key: {masked_key} (index: {self.index}/{len(self.keys)})")
            return key
