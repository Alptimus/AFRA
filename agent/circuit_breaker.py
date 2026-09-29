"""
Circuit Breaker — Prevents cascading failures when multiple tools fail.

Implements the Circuit Breaker pattern with three states:
- CLOSED: Normal operation, calls pass through.
- OPEN: Tool is failing, block calls immediately.
- HALF_OPEN: Testing if tool has recovered, allow one trial call.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"       # Normal — calls pass through
    OPEN = "open"           # Failing — calls blocked
    HALF_OPEN = "half_open" # Testing — one trial call allowed


@dataclass
class CircuitBreakerState:
    """State tracking for a single tool's circuit breaker.

    Attributes:
        state: Current circuit state.
        failure_count: Consecutive failure count.
        last_failure_time: Timestamp of the most recent failure.
        last_success_time: Timestamp of the most recent success.
        total_failures: Lifetime failure count.
        total_successes: Lifetime success count.
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """Circuit breaker pattern for agent tools.

    Prevents cascading failures by tracking consecutive failures per tool
    and temporarily blocking calls to failing tools.

    Configuration:
        failure_threshold: Consecutive failures before opening circuit (default 3).
        recovery_timeout: Seconds to wait before testing recovery (default 60).

    Usage:
        breaker = CircuitBreaker()

        if breaker.can_execute("web_search"):
            result = await tool.execute(...)
            if result.success:
                breaker.record_success("web_search")
            else:
                breaker.record_failure("web_search")
        else:
            # Tool is circuit-broken, use fallback
            ...
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._states: dict[str, CircuitBreakerState] = {}

    def _get_state(self, tool_name: str) -> CircuitBreakerState:
        """Get or create state for a tool."""
        if tool_name not in self._states:
            self._states[tool_name] = CircuitBreakerState()
        return self._states[tool_name]

    def can_execute(self, tool_name: str) -> bool:
        """Check if a tool call is allowed.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the call is allowed, False if circuit is open.
        """
        state = self._get_state(tool_name)

        if state.state == CircuitState.CLOSED:
            return True

        if state.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            elapsed = time.time() - state.last_failure_time
            if elapsed >= self._recovery_timeout:
                logger.info(
                    "Circuit breaker for %s transitioning OPEN → HALF_OPEN "
                    "(%.1fs since last failure)",
                    tool_name,
                    elapsed,
                )
                state.state = CircuitState.HALF_OPEN
                return True  # Allow one trial call
            return False

        # HALF_OPEN — allow the trial call
        return True

    def record_success(self, tool_name: str) -> None:
        """Record a successful tool execution.

        If the circuit was HALF_OPEN, this closes it (recovery confirmed).
        """
        state = self._get_state(tool_name)
        state.total_successes += 1
        state.last_success_time = time.time()

        if state.state == CircuitState.HALF_OPEN:
            logger.info(
                "Circuit breaker for %s transitioning HALF_OPEN → CLOSED (recovered)",
                tool_name,
            )
            state.state = CircuitState.CLOSED
            state.failure_count = 0

        elif state.state == CircuitState.CLOSED:
            state.failure_count = 0  # Reset consecutive failures

    def record_failure(self, tool_name: str) -> None:
        """Record a failed tool execution.

        If consecutive failures reach the threshold, opens the circuit.
        If the circuit was HALF_OPEN, reopens it (recovery failed).
        """
        state = self._get_state(tool_name)
        state.failure_count += 1
        state.total_failures += 1
        state.last_failure_time = time.time()

        if state.state == CircuitState.HALF_OPEN:
            logger.warning(
                "Circuit breaker for %s transitioning HALF_OPEN → OPEN "
                "(trial call failed)",
                tool_name,
            )
            state.state = CircuitState.OPEN

        elif state.state == CircuitState.CLOSED:
            if state.failure_count >= self._failure_threshold:
                logger.warning(
                    "Circuit breaker for %s transitioning CLOSED → OPEN "
                    "(%d consecutive failures)",
                    tool_name,
                    state.failure_count,
                )
                state.state = CircuitState.OPEN

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all tracked circuit breakers."""
        return {
            name: {
                "state": state.state.value,
                "failure_count": state.failure_count,
                "total_failures": state.total_failures,
                "total_successes": state.total_successes,
            }
            for name, state in self._states.items()
        }

    def reset(self, tool_name: str | None = None) -> None:
        """Reset circuit breaker state.

        Args:
            tool_name: Specific tool to reset, or None to reset all.
        """
        if tool_name:
            self._states.pop(tool_name, None)
        else:
            self._states.clear()
