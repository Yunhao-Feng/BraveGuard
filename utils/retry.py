"""通用重试装饰器和辅助函数"""
import asyncio
import logging
from functools import wraps
from typing import TypeVar, Callable, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')


def async_retry(
    max_retries: int = 3,
    delay: float = 2.0,
    exceptions: tuple = (Exception,),
    backoff: float = 1.0
):
    """
    异步重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟秒数
        exceptions: 需要重试的异常类型
        backoff: 延迟倍增因子
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} 最终失败 (尝试 {max_retries} 次): {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


async def safe_execute(coro: Any, default: Any = None, log_error: bool = True) -> Any:
    """
    安全执行协程，捕获异常并返回默认值

    Args:
        coro: 协程对象
        default: 异常时返回的默认值
        log_error: 是否记录错误日志
    """
    try:
        return await coro
    except Exception as e:
        if log_error:
            logger.error(f"执行失败: {e}", exc_info=True)
        return default
