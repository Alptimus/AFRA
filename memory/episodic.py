"""
Episodic Memory — Records and learns from the agent's past research experiences.

Tracks which research strategies worked well, which tools produced the best
results for specific query types, error patterns, and successful recovery
strategies. This enables the agent to improve its planning over time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ResearchEpisode:
    """A recorded research experience.

    Attributes:
        session_id: Unique identifier for the research session.
        query: The original research query.
        query_type: Classified query type.
        strategy: The research strategy/plan used.
        tools_used: List of tools called and their success rates.
        outcome: Quality assessment of the final output.
        errors: Errors encountered and how they were handled.
        duration_seconds: Total research time.
        timestamp: When the episode occurred.
        notes: Any additional observations about the experience.
    """

    session_id: str = ""
    query: str = ""
    query_type: str = ""
    strategy: list[str] = field(default_factory=list)
    tools_used: dict[str, dict[str, Any]] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class StrategyRecommendation:
    """A recommended research strategy based on past episodes.

    Attributes:
        recommended_tools: Ordered list of tools to use.
        tool_order: Suggested order of tool calls.
        estimated_calls: Expected number of tool calls.
        confidence: Confidence in this recommendation (0-1).
        based_on_episodes: Number of past episodes informing this recommendation.
        notes: Strategy notes from past experience.
    """

    recommended_tools: list[str] = field(default_factory=list)
    tool_order: list[str] = field(default_factory=list)
    estimated_calls: int = 10
    confidence: float = 0.5
    based_on_episodes: int = 0
    notes: str = ""


class EpisodicMemory:
    """Records and learns from the agent's research experiences.

    Stores episodes as JSON files in the data directory. On future queries,
    retrieves the most relevant past episodes to inform research planning.

    Usage:
        memory = EpisodicMemory()

        # Record a completed research session
        memory.record_episode(episode)

        # Get strategy recommendation for a new query
        recommendation = memory.get_best_strategy("risk_assessment")
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._episodes_dir = self._settings.data_dir / "episodic_memory"
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        self._episodes: list[ResearchEpisode] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load episodes from disk on first access."""
        if self._loaded:
            return

        for path in self._episodes_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                self._episodes.append(ResearchEpisode(**data))
            except Exception as e:
                logger.warning("Failed to load episode %s: %s", path, e)

        self._loaded = True
        logger.info("Loaded %d episodic memories", len(self._episodes))

    def record_episode(self, episode: ResearchEpisode) -> None:
        """Record a research episode to persistent storage.

        Args:
            episode: The completed research episode to record.
        """
        self._ensure_loaded()

        if not episode.session_id:
            episode.session_id = f"session-{int(episode.timestamp)}"

        # Save to disk
        filename = f"{episode.session_id}.json"
        filepath = self._episodes_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(asdict(episode), f, indent=2, default=str)
            self._episodes.append(episode)
            logger.info("Recorded episode: %s", episode.session_id)
        except Exception as e:
            logger.error("Failed to save episode: %s", e)

    def get_best_strategy(self, query_type: str) -> StrategyRecommendation:
        """Get the best research strategy for a given query type.

        Analyzes past episodes of the same query type to determine
        which tools and strategies produced the best outcomes.

        Args:
            query_type: The classified query type (e.g., "risk_assessment").

        Returns:
            StrategyRecommendation based on past experience.
        """
        self._ensure_loaded()

        # Filter episodes by query type
        relevant = [
            ep for ep in self._episodes if ep.query_type == query_type
        ]

        if not relevant:
            return StrategyRecommendation(
                notes=f"No past episodes for query type '{query_type}'",
            )

        # Analyze tool effectiveness across relevant episodes
        tool_success: dict[str, list[float]] = {}
        for ep in relevant:
            for tool_name, stats in ep.tools_used.items():
                if tool_name not in tool_success:
                    tool_success[tool_name] = []
                success_rate = stats.get("success_rate", 0.5)
                tool_success[tool_name].append(success_rate)

        # Rank tools by average success rate
        tool_ranking = sorted(
            tool_success.items(),
            key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
            reverse=True,
        )

        recommended = [name for name, _ in tool_ranking]
        avg_calls = (
            sum(sum(len(ep.tools_used) for _ in [1]) for ep in relevant)
            / len(relevant)
            if relevant
            else 10
        )

        return StrategyRecommendation(
            recommended_tools=recommended,
            tool_order=recommended,
            estimated_calls=int(avg_calls),
            confidence=min(0.9, 0.3 + 0.1 * len(relevant)),
            based_on_episodes=len(relevant),
            notes=(
                f"Based on {len(relevant)} past episodes. "
                f"Top tools: {', '.join(recommended[:5])}"
            ),
        )

    def get_error_patterns(self) -> dict[str, list[str]]:
        """Identify common error patterns across all episodes.

        Returns:
            Dict mapping error types to lists of descriptions.
        """
        self._ensure_loaded()
        patterns: dict[str, list[str]] = {}

        for ep in self._episodes:
            for error in ep.errors:
                error_type = error.get("category", "unknown")
                message = error.get("message", "")
                if error_type not in patterns:
                    patterns[error_type] = []
                if message and message not in patterns[error_type]:
                    patterns[error_type].append(message)

        return patterns

    @property
    def episode_count(self) -> int:
        """Number of recorded episodes."""
        self._ensure_loaded()
        return len(self._episodes)
