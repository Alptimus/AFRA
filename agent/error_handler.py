"""
Error Handler — Comprehensive error handling with retry and graceful degradation.

Implements retry with exponential backoff, error classification, and graceful
degradation protocols for the financial research agent.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"           # Cosmetic or non-critical
    MEDIUM = "medium"     # Degraded but functional
    HIGH = "high"         # Major functionality impacted
    CRITICAL = "critical" # Cannot continue


class ErrorCategory(Enum):
    """Categories of agent errors."""

    TOOL_EXECUTION = "tool_execution"       # API errors, timeouts, rate limits
    REASONING = "reasoning"                 # Hallucination, logical errors
    DATA_QUALITY = "data_quality"           # Stale data, conflicts, misattribution
    CONFIGURATION = "configuration"         # Missing API keys, bad config
    NETWORK = "network"                     # Connection errors, DNS, SSL


@dataclass
class AgentError:
    """Structured error record.

    Attributes:
        category: Classification of the error.
        severity: How severe the error is.
        message: Human-readable error description.
        tool_name: Which tool was involved (if applicable).
        recoverable: Whether the error can potentially be recovered from.
        recovery_action: Description of recovery action taken (if any).
        timestamp: When the error occurred.
    """

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    tool_name: str = ""
    recoverable: bool = True
    recovery_action: str = ""
    timestamp: float = field(default_factory=time.time)


class ErrorHandler:
    """Manages error handling, retry logic, and graceful degradation.

    Strategies:
    1. Retry with exponential backoff (1s base, max 5 retries, 0-500ms jitter)
    2. Fallback tool chains (delegated to ToolRegistry)
    3. Graceful degradation (transparent failure reporting in output)

    Usage:
        handler = ErrorHandler(max_retries=5)

        # Retry a function
        result = await handler.retry_with_backoff(some_async_func, arg1, arg2)

        # Record an error
        handler.record_error(AgentError(...))

        # Get degradation report for final output
        report = handler.get_degradation_report()
    """

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._errors: list[AgentError] = []
        self._recovery_count: int = 0

    async def retry_with_backoff(
        self,
        func: Callable,
        *args: Any,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with exponential backoff retry.

        Implements the retry strategy from Section A4.3:
        - Initial delay: 1 second
        - Doubles with each attempt
        - Max 5 retries (configurable)
        - Random jitter: 0-500ms per attempt

        Args:
            func: Async callable to retry.
            *args: Positional arguments for the function.
            max_retries: Override the default max retry count.
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        retries = max_retries or self._max_retries
        last_exception = None

        for attempt in range(retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == retries:
                    logger.error(
                        "All %d retries exhausted for %s: %s",
                        retries,
                        getattr(func, "__name__", "unknown"),
                        e,
                    )
                    break

                # Exponential backoff with jitter
                delay = self._base_delay * (2 ** attempt)
                jitter = random.uniform(0, 0.5)
                total_delay = delay + jitter

                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.2fs",
                    attempt + 1,
                    retries,
                    getattr(func, "__name__", "unknown"),
                    e,
                    total_delay,
                )

                import asyncio
                await asyncio.sleep(total_delay)

        raise last_exception  # type: ignore[misc]

    def record_error(self, error: AgentError) -> None:
        """Record an error for tracking and reporting."""
        self._errors.append(error)
        logger.log(
            logging.ERROR if error.severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL) else logging.WARNING,
            "[%s] %s: %s (tool=%s, recoverable=%s)",
            error.severity.value,
            error.category.value,
            error.message,
            error.tool_name,
            error.recoverable,
        )

    def record_recovery(self, error: AgentError, recovery_action: str) -> None:
        """Record that an error was successfully recovered from."""
        error.recovery_action = recovery_action
        self._recovery_count += 1
        logger.info(
            "Recovered from %s error on %s: %s",
            error.category.value,
            error.tool_name,
            recovery_action,
        )

    def classify_error(self, exception: Exception, tool_name: str = "") -> AgentError:
        """Classify an exception into a structured AgentError.

        Args:
            exception: The caught exception.
            tool_name: Name of the tool that raised the error.

        Returns:
            Classified AgentError.
        """
        import httpx

        error_type = type(exception).__name__
        message = str(exception)

        # Network / HTTP errors
        if isinstance(exception, httpx.HTTPStatusError):
            status = exception.response.status_code
            if status == 429:
                return AgentError(
                    category=ErrorCategory.TOOL_EXECUTION,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Rate limited (HTTP 429): {message}",
                    tool_name=tool_name,
                    recoverable=True,
                )
            elif status in (500, 502, 503, 504):
                return AgentError(
                    category=ErrorCategory.TOOL_EXECUTION,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Server error (HTTP {status}): {message}",
                    tool_name=tool_name,
                    recoverable=True,
                )
            elif status in (401, 403):
                return AgentError(
                    category=ErrorCategory.CONFIGURATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Authentication error (HTTP {status}): {message}",
                    tool_name=tool_name,
                    recoverable=False,
                )
            else:
                return AgentError(
                    category=ErrorCategory.TOOL_EXECUTION,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"HTTP error ({status}): {message}",
                    tool_name=tool_name,
                    recoverable=True,
                )

        if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException)):
            return AgentError(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                message=f"Network error: {message}",
                tool_name=tool_name,
                recoverable=True,
            )

        # Data errors
        if isinstance(exception, (ValueError, KeyError)):
            return AgentError(
                category=ErrorCategory.DATA_QUALITY,
                severity=ErrorSeverity.LOW,
                message=f"Data error: {message}",
                tool_name=tool_name,
                recoverable=True,
            )

        # Generic
        return AgentError(
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            message=f"{error_type}: {message}",
            tool_name=tool_name,
            recoverable=True,
        )

    # ── Degradation Reporting ─────────────────────────────────────────────

    def get_degradation_report(self) -> dict[str, Any]:
        """Generate a report of all errors for inclusion in the final output.

        This implements the Graceful Degradation Protocol from Section A4.3:
        - Clearly state which sections could not be completed and why
        - Provide the best available information from tools that did work
        - Suggest alternatives for missing information
        - Never fabricate data to fill gaps
        """
        if not self._errors:
            return {"has_degradation": False, "errors": []}

        unrecovered = [e for e in self._errors if not e.recovery_action]
        recovered = [e for e in self._errors if e.recovery_action]

        return {
            "has_degradation": bool(unrecovered),
            "total_errors": len(self._errors),
            "recovered_errors": len(recovered),
            "unrecovered_errors": len(unrecovered),
            "recovery_rate": (
                self._recovery_count / len(self._errors)
                if self._errors
                else 1.0
            ),
            "errors": [
                {
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "message": e.message,
                    "tool": e.tool_name,
                    "recovered": bool(e.recovery_action),
                    "recovery_action": e.recovery_action,
                }
                for e in self._errors
            ],
            "affected_tools": list(set(e.tool_name for e in unrecovered if e.tool_name)),
            "degradation_note": (
                "The following data sources were unavailable during research: "
                + ", ".join(set(e.tool_name for e in unrecovered if e.tool_name))
                + ". Sections relying on these sources may be incomplete."
                if unrecovered
                else ""
            ),
        }

    @property
    def error_count(self) -> int:
        """Total number of errors recorded."""
        return len(self._errors)

    @property
    def recovery_rate(self) -> float:
        """Percentage of errors successfully recovered from."""
        if not self._errors:
            return 1.0
        return self._recovery_count / len(self._errors)

    def reset(self) -> None:
        """Reset error tracking (between challenges/tasks)."""
        self._errors.clear()
        self._recovery_count = 0
