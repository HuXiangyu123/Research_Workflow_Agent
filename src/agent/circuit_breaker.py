"""熔断器实现 — 保护 agent 系统免受级联故障影响。

设计基于 Martin Fowler CircuitBreaker 模式：
- CLOSED → OPEN：连续失败达到 threshold
- OPEN → HALF_OPEN：timeout_s 冷却时间结束
- HALF_OPEN → CLOSED：成功率达到 success_threshold
- HALF_OPEN → OPEN：任何一个测试请求失败

使用 threading.Lock 保证并发安全。
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"       # 正常：请求通过
    OPEN = "open"            # 熔断：请求被拒绝或返回降级值
    HALF_OPEN = "half_open"  # 半开：放行测试请求


@dataclass
class CircuitBreakerConfig:
    """熔断器配置参数。"""

    failure_threshold: int = 5       # 连续失败多少次后打开熔断器
    timeout_s: float = 60.0         # 熔断器打开后多少秒进入半开状态
    half_open_max_calls: int = 1    # 半开状态下最多放行多少个测试请求
    success_threshold: int = 1     # 半开状态下成功多少次才关闭熔断器


class CircuitBreaker:
    """
    熔断器：保护外部调用免受级联故障影响。

    状态转换：
    CLOSED → OPEN：连续失败达到 failure_threshold
    OPEN → HALF_OPEN：timeout_s 冷却时间结束
    HALF_OPEN → CLOSED：成功率达到 success_threshold
    HALF_OPEN → OPEN：任何一个测试请求失败
    """

    def __init__(self, key: str, config: CircuitBreakerConfig | None = None):
        self.key = key
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查冷却时间
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.config.timeout_s:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        self._success_count = 0
                        logger.info("[CircuitBreaker] %s OPEN → HALF_OPEN (cooled down after %.1fs)", self.key, elapsed)
            return self._state

    def can_execute(self) -> bool:
        """判断是否可以执行请求。"""
        s = self.state
        if s == CircuitState.CLOSED:
            return True
        if s == CircuitState.OPEN:
            return False
        # HALF_OPEN
        with self._lock:
            return self._half_open_calls < self.config.half_open_max_calls

    def record_success(self) -> None:
        """记录一次成功调用。"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    logger.info("[CircuitBreaker] %s HALF_OPEN → CLOSED (success threshold reached)", self.key)
            else:
                self._failure_count = 0

    def record_failure(self) -> None:
        """记录一次失败调用。"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                self._success_count = 0
                logger.warning("[CircuitBreaker] %s HALF_OPEN → OPEN (test request failed)", self.key)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "[CircuitBreaker] %s CLOSED → OPEN (failures=%d/%d)",
                        self.key,
                        self._failure_count,
                        self.config.failure_threshold,
                    )

    def record_half_open_call(self) -> None:
        """记录一次半开状态的测试调用（用于计数限制）。"""
        with self._lock:
            self._half_open_calls += 1

    def execute(
        self,
        func: Callable[..., T],
        fallback: Callable[[], T] | T | None = None,
        *args,
        **kwargs,
    ) -> T:
        """
        通过熔断器执行调用。

        Args:
            func: 要执行的函数
            fallback: 熔断打开时的降级函数（可选）
            *args, **kwargs: 传递给 func 的参数

        Returns:
            func() 的结果，或 fallback() 的结果
        """
        if not self.can_execute():
            if fallback is not None:
                logger.warning("[CircuitBreaker] %s is OPEN, using fallback", self.key)
                if callable(fallback):
                    return fallback()
                return fallback
            raise CircuitOpenError(f"Circuit breaker '{self.key}' is OPEN")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure()
            if fallback is not None:
                logger.warning(
                    "[CircuitBreaker] %s call failed (%s: %s), using fallback",
                    self.key,
                    type(exc).__name__,
                    exc,
                )
                if callable(fallback):
                    return fallback()
                return fallback
            raise


class CircuitOpenError(Exception):
    """熔断器打开时抛出的异常。"""

    pass


# ---------------------------------------------------------------------------
# 全局熔断器注册表
# ---------------------------------------------------------------------------


@dataclass
class BreakerSpec:
    """熔断器规格。"""

    key: str
    config: CircuitBreakerConfig


# 全局注册表
_BREAKER_REGISTRY: dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()

# 默认配置（按 API 类型）
_DEFAULT_CONFIGS: dict[str, CircuitBreakerConfig] = {
    "deepseek/deepseek-chat": CircuitBreakerConfig(failure_threshold=5, timeout_s=60.0),
    "deepseek/deepseek-reasoner": CircuitBreakerConfig(failure_threshold=3, timeout_s=30.0),
    "openai/gpt-4o": CircuitBreakerConfig(failure_threshold=5, timeout_s=60.0),
    "openai/gpt-4o-mini": CircuitBreakerConfig(failure_threshold=5, timeout_s=60.0),
    "openai/gpt-5.4": CircuitBreakerConfig(failure_threshold=5, timeout_s=60.0),
    "qwen/text-embedding-v4": CircuitBreakerConfig(failure_threshold=3, timeout_s=30.0),
    "searxng": CircuitBreakerConfig(failure_threshold=3, timeout_s=60.0),
    "arxiv": CircuitBreakerConfig(failure_threshold=5, timeout_s=120.0),
    "arxiv/direct": CircuitBreakerConfig(failure_threshold=5, timeout_s=180.0),
    "http/download": CircuitBreakerConfig(failure_threshold=3, timeout_s=30.0),
}


def get_breaker(
    provider: str,
    model: str | None = None,
    endpoint: str | None = None,
) -> CircuitBreaker:
    """
    获取或创建指定 provider 的熔断器。

    key 格式："{provider}" 或 "{provider}/{model}" 或 "{provider}/{model}/{endpoint}"
    """
    key_parts = [provider]
    if model:
        key_parts.append(model)
    if endpoint:
        key_parts.append(endpoint)
    key = "/".join(key_parts)

    with _REGISTRY_LOCK:
        if key not in _BREAKER_REGISTRY:
            # 精确匹配优先，其次前缀匹配
            config = _DEFAULT_CONFIGS.get(key)
            if config is None and model:
                # 尝试 provider/model 前缀
                config = _DEFAULT_CONFIGS.get(f"{provider}/{model}")
            if config is None:
                # 尝试纯 provider
                config = _DEFAULT_CONFIGS.get(provider)
            if config is None:
                config = CircuitBreakerConfig()  # 默认配置

            _BREAKER_REGISTRY[key] = CircuitBreaker(key=key, config=config)
            logger.debug("[CircuitBreaker] Registered: %s (threshold=%d, timeout=%.0fs)", key, config.failure_threshold, config.timeout_s)

        return _BREAKER_REGISTRY[key]


def get_all_breaker_status() -> dict[str, dict]:
    """获取所有熔断器当前状态（用于监控和 API）。"""
    with _REGISTRY_LOCK:
        return {
            key: {
                "state": cb.state.value,
                "failure_count": cb._failure_count,
                "success_count": cb._success_count,
                "config": {
                    "failure_threshold": cb.config.failure_threshold,
                    "timeout_s": cb.config.timeout_s,
                    "half_open_max_calls": cb.config.half_open_max_calls,
                    "success_threshold": cb.config.success_threshold,
                },
            }
            for key, cb in _BREAKER_REGISTRY.items()
        }


def reset_breaker(key: str) -> None:
    """重置指定熔断器（用于测试或人工干预）。"""
    with _REGISTRY_LOCK:
        if key in _BREAKER_REGISTRY:
            cb = _BREAKER_REGISTRY[key]
            with cb._lock:
                cb._state = CircuitState.CLOSED
                cb._failure_count = 0
                cb._success_count = 0
                cb._half_open_calls = 0
                cb._last_failure_time = None
            logger.info("[CircuitBreaker] %s manually reset to CLOSED", key)


def reset_all_breakers() -> None:
    """重置所有熔断器（用于测试）。"""
    with _REGISTRY_LOCK:
        for key in list(_BREAKER_REGISTRY.keys()):
            reset_breaker(key)
