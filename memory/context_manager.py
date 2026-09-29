"""
Context Manager — Short-term memory management for the agent.

Manages the agent's working memory within LLM context window limits.
Implements progressive summarization and token budgeting strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """A single observation from a tool call or reasoning step.

    Attributes:
        source: Tool name or "reasoning".
        content: The observation text.
        token_estimate: Rough token count.
        importance: Importance score (0-1, higher = more important to retain).
        step: Which research step produced this observation.
    """

    source: str
    content: str
    token_estimate: int = 0
    importance: float = 0.5
    step: int = 0


class ContextManager:
    """Manages the agent's short-term working memory.

    Strategies:
    1. Token budgeting: 40% primary data, 30% supporting, 20% system, 10% generation.
    2. Progressive summarization: Compress old observations when near limit.
    3. Selective retention: Keep high-importance findings, compress low-importance ones.

    Usage:
        ctx = ContextManager(max_tokens=100000)
        ctx.add_observation(Observation(source="sec_filing_search", content="..."))
        context_str = ctx.get_context()
    """

    # Token budget allocation (Section A8.2, Stage 4)
    BUDGET_PRIMARY = 0.40      # SEC filings, financial data
    BUDGET_SUPPORTING = 0.30   # News, analyst reports
    BUDGET_SYSTEM = 0.20       # System prompt, tool descriptions
    BUDGET_GENERATION = 0.10   # Reserved for LLM output

    def __init__(self, max_tokens: int = 100000) -> None:
        self._max_tokens = max_tokens
        self._observations: list[Observation] = []
        self._summaries: list[str] = []
        self._total_tokens: int = 0
        self._available_tokens = int(
            max_tokens * (1 - self.BUDGET_SYSTEM - self.BUDGET_GENERATION)
        )

    def add_observation(self, observation: Observation) -> None:
        """Add a new observation to working memory.

        If adding this observation would exceed the token budget,
        older low-importance observations are summarized first.

        Args:
            observation: The observation to add.
        """
        if not observation.token_estimate:
            observation.token_estimate = self._estimate_tokens(observation.content)

        # Check if we need to compress
        if self._total_tokens + observation.token_estimate > self._available_tokens:
            self._compress()

        self._observations.append(observation)
        self._total_tokens += observation.token_estimate

    def get_context(self) -> str:
        """Build the context string for the LLM.

        Returns observations in chronological order with source labels.
        Includes any compressed summaries at the top.
        """
        parts = []

        # Include compressed summaries first
        if self._summaries:
            parts.append("=== PREVIOUS RESEARCH SUMMARY ===")
            parts.extend(self._summaries)
            parts.append("=== END SUMMARY ===\n")

        # Include current observations
        for obs in self._observations:
            parts.append(f"[Step {obs.step} | Source: {obs.source}]")
            parts.append(obs.content)
            parts.append("")

        return "\n".join(parts)

    def get_key_findings(self) -> list[str]:
        """Extract high-importance findings for report generation."""
        findings = []
        for obs in self._observations:
            if obs.importance >= 0.7:
                findings.append(f"[{obs.source}] {obs.content[:500]}")
        return findings

    @property
    def token_usage(self) -> dict[str, int]:
        """Current token usage breakdown."""
        return {
            "total_tokens": self._total_tokens,
            "available_tokens": self._available_tokens,
            "max_tokens": self._max_tokens,
            "observations": len(self._observations),
            "summaries": len(self._summaries),
            "utilization_pct": round(
                self._total_tokens / self._available_tokens * 100, 1
            )
            if self._available_tokens
            else 0,
        }

    def _compress(self) -> None:
        """Compress older, low-importance observations into summaries.

        Keeps the most recent and highest-importance observations intact.
        Older, lower-importance observations are compressed into a text summary.
        """
        if len(self._observations) < 3:
            return  # Not enough to compress

        # Sort by importance, keep top half
        sorted_obs = sorted(
            self._observations, key=lambda o: o.importance, reverse=True
        )
        keep_count = max(len(sorted_obs) // 2, 2)
        to_keep = sorted_obs[:keep_count]
        to_compress = sorted_obs[keep_count:]

        if not to_compress:
            return

        # Generate summary of compressed observations
        summary_parts = ["Key findings from earlier research steps:"]
        for obs in to_compress:
            # Keep first 200 chars of each compressed observation
            snippet = obs.content[:200].replace('\n', ' ').strip()
            summary_parts.append(f"- [{obs.source}] {snippet}")

        self._summaries.append("\n".join(summary_parts))

        # Reset observations to only kept ones (in original order)
        kept_indices = {id(o) for o in to_keep}
        self._observations = [o for o in self._observations if id(o) in kept_indices]
        self._total_tokens = sum(o.token_estimate for o in self._observations)

        logger.debug(
            "Compressed context: kept %d obs, compressed %d, tokens now %d",
            len(to_keep),
            len(to_compress),
            self._total_tokens,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return max(1, len(text) // 4)

    def reset(self) -> None:
        """Clear all working memory."""
        self._observations.clear()
        self._summaries.clear()
        self._total_tokens = 0
