"""
Tool Registry — Central catalog of all AFRA agent tools.

The registry serves three purposes:
1. Provides the LLM with tool metadata for intelligent selection.
2. Routes tool calls to the correct implementations.
3. Manages rate limiting, caching, error handling, and fallback chains.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config.settings import Settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# ── Rate Limiter ──────────────────────────────────────────────────────────────


@dataclass
class RateLimiter:
    """Simple token-bucket rate limiter for API tools.

    Attributes:
        max_calls: Maximum number of calls allowed in the time window.
        window_seconds: Time window in seconds.
    """

    max_calls: int = 10
    window_seconds: float = 60.0
    _calls: list[float] = field(default_factory=list)

    def acquire(self) -> bool:
        """Try to acquire a rate limit token.

        Returns:
            True if the call is allowed, False if rate-limited.
        """
        now = time.time()
        # Remove expired entries
        self._calls = [t for t in self._calls if now - t < self.window_seconds]
        if len(self._calls) >= self.max_calls:
            return False
        self._calls.append(now)
        return True

    @property
    def wait_time(self) -> float:
        """Seconds until the next call is allowed (0 if not rate-limited)."""
        if not self._calls or len(self._calls) < self.max_calls:
            return 0.0
        oldest = min(self._calls)
        return max(0.0, self.window_seconds - (time.time() - oldest))


# ── Cache Entry ───────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """Cached tool result with TTL.

    Attributes:
        result: The cached ToolResult.
        timestamp: When the result was cached.
        ttl_seconds: Time-to-live in seconds.
    """

    result: ToolResult
    timestamp: float
    ttl_seconds: float = 300.0  # 5 minutes default

    @property
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return (time.time() - self.timestamp) > self.ttl_seconds


# ── Tool Registry ─────────────────────────────────────────────────────────────


class ToolRegistry:
    """Central catalog of all agent tools.

    Manages tool registration, discovery, execution routing, rate limiting,
    caching, and fallback chain resolution.

    Usage:
        registry = ToolRegistry(settings)
        registry.register(SecFilingSearchTool())
        registry.register(WebSearchTool())

        # Get schemas for LLM
        schemas = registry.get_tool_schemas(format="openai")

        # Execute a tool call
        result = await registry.execute("web_search", query="Tesla risks 2024")
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tools: dict[str, BaseTool] = {}
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._cache: dict[str, CacheEntry] = {}
        self._call_log: list[ToolCallRecord] = []

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, tool: BaseTool, rate_limit: RateLimiter | None = None) -> None:
        """Register a tool in the registry.

        Args:
            tool: The tool instance to register.
            rate_limit: Optional rate limiter for this tool.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        if rate_limit:
            self._rate_limiters[tool.name] = rate_limit
        logger.info("Registered tool: %s", tool.name)

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(tool_name, None)
        self._rate_limiters.pop(tool_name, None)

    # ── Discovery ─────────────────────────────────────────────────────────

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool instance by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())

    def get_tool_schemas(
        self, format: str = "openai"
    ) -> list[dict[str, Any]]:
        """Get all tool schemas in the specified LLM format.

        Args:
            format: Schema format — "openai" or "anthropic".

        Returns:
            List of tool schema dictionaries ready for LLM injection.
        """
        schemas = []
        for tool in self._tools.values():
            schema = tool.get_schema()
            if format == "anthropic":
                schemas.append(schema.to_anthropic_format())
            else:
                schemas.append(schema.to_openai_format())
        return schemas

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(
        self,
        tool_name: str,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool by name with rate limiting, caching, and logging.

        Args:
            tool_name: Name of the tool to execute.
            use_cache: Whether to check/store cache.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult from the tool or fallback chain.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Unknown tool: '{tool_name}'",
                source="registry",
            )

        # Check cache
        if use_cache:
            cached = self._check_cache(tool_name, kwargs)
            if cached:
                logger.debug("Cache hit for %s", tool_name)
                return cached

        # Check rate limit
        limiter = self._rate_limiters.get(tool_name)
        if limiter and not limiter.acquire():
            wait = limiter.wait_time
            logger.warning(
                "Rate limited on %s, wait %.1fs", tool_name, wait
            )
            return ToolResult(
                success=False,
                error=f"Rate limited. Retry after {wait:.1f}s",
                source=tool_name,
                metadata={"rate_limited": True, "retry_after_seconds": wait},
            )

        # Execute
        result = await tool.execute(**kwargs)

        # Log the call
        self._call_log.append(
            ToolCallRecord(
                tool_name=tool_name,
                kwargs=kwargs,
                success=result.success,
                execution_time_ms=result.execution_time_ms,
                timestamp=time.time(),
            )
        )

        # Cache successful results
        if result.success and use_cache:
            self._store_cache(tool_name, kwargs, result)

        return result

    async def execute_with_fallback(
        self, tool_name: str, **kwargs: Any
    ) -> ToolResult:
        """Execute a tool, falling back to alternatives on failure.

        Tries the primary tool first, then iterates through the tool's
        fallback_tools list until one succeeds.

        Args:
            tool_name: Primary tool name.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult from the first successful tool in the chain.
        """
        # Try primary tool
        result = await self.execute(tool_name, **kwargs)
        if result.success:
            return result

        # Get fallback chain
        primary_tool = self._tools.get(tool_name)
        if not primary_tool or not primary_tool.fallback_tools:
            return result

        logger.warning(
            "Tool %s failed (%s), trying fallback chain: %s",
            tool_name,
            result.error,
            primary_tool.fallback_tools,
        )

        # Try each fallback
        for fallback_name in primary_tool.fallback_tools:
            fallback_result = await self.execute(fallback_name, **kwargs)
            if fallback_result.success:
                fallback_result.metadata["fallback_from"] = tool_name
                fallback_result.metadata["fallback_tool"] = fallback_name
                logger.info(
                    "Fallback succeeded: %s → %s", tool_name, fallback_name
                )
                return fallback_result
            logger.warning("Fallback %s also failed: %s", fallback_name, fallback_result.error)

        # All fallbacks failed
        return ToolResult(
            success=False,
            error=(
                f"Tool '{tool_name}' and all fallbacks "
                f"({', '.join(primary_tool.fallback_tools)}) failed. "
                f"Last error: {result.error}"
            ),
            source=tool_name,
            metadata={"all_fallbacks_exhausted": True},
        )

    # ── Cache Management ──────────────────────────────────────────────────

    def _cache_key(self, tool_name: str, kwargs: dict) -> str:
        """Generate a deterministic cache key."""
        import hashlib
        import json as json_mod

        payload = json_mod.dumps(
            {"tool": tool_name, "args": kwargs}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _check_cache(self, tool_name: str, kwargs: dict) -> ToolResult | None:
        """Check if a valid cached result exists."""
        key = self._cache_key(tool_name, kwargs)
        entry = self._cache.get(key)
        if entry and not entry.is_expired:
            return entry.result
        if entry and entry.is_expired:
            del self._cache[key]
        return None

    def _store_cache(
        self, tool_name: str, kwargs: dict, result: ToolResult
    ) -> None:
        """Store a result in the cache."""
        key = self._cache_key(tool_name, kwargs)
        self._cache[key] = CacheEntry(result=result, timestamp=time.time())

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    # ── Telemetry ─────────────────────────────────────────────────────────

    @property
    def call_log(self) -> list[ToolCallRecord]:
        """Return the full call log for evaluation metrics."""
        return self._call_log

    @property
    def total_calls(self) -> int:
        """Total number of tool calls made."""
        return len(self._call_log)

    @property
    def successful_calls(self) -> int:
        """Number of successful tool calls."""
        return sum(1 for r in self._call_log if r.success)

    def get_tool_stats(self) -> dict[str, dict[str, Any]]:
        """Get per-tool execution statistics.

        Returns:
            Dict mapping tool name to stats (call_count, success_rate, avg_time_ms).
        """
        from collections import defaultdict

        stats: dict[str, list[ToolCallRecord]] = defaultdict(list)
        for record in self._call_log:
            stats[record.tool_name].append(record)

        result = {}
        for name, records in stats.items():
            successes = sum(1 for r in records if r.success)
            avg_time = sum(r.execution_time_ms for r in records) / len(records)
            result[name] = {
                "call_count": len(records),
                "success_rate": successes / len(records) if records else 0.0,
                "avg_time_ms": round(avg_time, 2),
            }
        return result

    def reset_stats(self) -> None:
        """Reset the call log (for testing or between challenges)."""
        self._call_log.clear()


@dataclass
class ToolCallRecord:
    """Record of a single tool call for telemetry."""

    tool_name: str
    kwargs: dict[str, Any]
    success: bool
    execution_time_ms: float
    timestamp: float
