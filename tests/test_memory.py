"""
Tests for the memory system — vector store, context manager, and episodic memory.
"""

from __future__ import annotations

import asyncio
import pytest
import tempfile
from pathlib import Path

from config.settings import Settings
from memory.context_manager import ContextManager, Observation
from memory.episodic import EpisodicMemory, ResearchEpisode


# ── Context Manager Tests ─────────────────────────────────────────────────────


class TestContextManager:
    """Tests for the short-term context manager."""

    def test_add_observation(self):
        """Test adding an observation to context."""
        ctx = ContextManager(max_tokens=100000)
        obs = Observation(source="test_tool", content="Test finding", importance=0.8, step=1)
        ctx.add_observation(obs)
        assert "Test finding" in ctx.get_context()

    def test_token_usage(self):
        """Test token usage tracking."""
        ctx = ContextManager(max_tokens=100000)
        obs = Observation(source="test", content="A" * 400, importance=0.5, step=1)
        ctx.add_observation(obs)
        usage = ctx.token_usage
        assert usage["observations"] == 1
        assert usage["total_tokens"] > 0

    def test_compression_triggers(self):
        """Test that compression triggers when approaching token limit."""
        ctx = ContextManager(max_tokens=500)  # Very small limit to force compression
        for i in range(10):
            obs = Observation(
                source=f"tool_{i}",
                content=f"Finding number {i}: " + "x" * 100,
                importance=0.1 * i,
                step=i,
            )
            ctx.add_observation(obs)

        # After compression, some observations should have been summarized
        context = ctx.get_context()
        assert context  # Should still have content

    def test_key_findings(self):
        """Test extracting high-importance findings."""
        ctx = ContextManager(max_tokens=100000)
        ctx.add_observation(Observation(source="a", content="Low importance", importance=0.2, step=1))
        ctx.add_observation(Observation(source="b", content="High importance", importance=0.9, step=2))
        ctx.add_observation(Observation(source="c", content="Medium", importance=0.5, step=3))

        findings = ctx.get_key_findings()
        assert len(findings) == 1  # Only importance >= 0.7
        assert "High importance" in findings[0]

    def test_reset(self):
        """Test context reset."""
        ctx = ContextManager(max_tokens=100000)
        ctx.add_observation(Observation(source="test", content="data", importance=0.5, step=1))
        ctx.reset()
        assert ctx.token_usage["observations"] == 0
        assert ctx.token_usage["total_tokens"] == 0


# ── Episodic Memory Tests ─────────────────────────────────────────────────────


class TestEpisodicMemory:
    """Tests for the episodic memory system."""

    @pytest.fixture
    def temp_settings(self, tmp_path):
        """Create settings with a temporary data directory."""
        return Settings(
            chroma_persist_dir=str(tmp_path / "chroma"),
        )

    @pytest.fixture
    def memory(self, temp_settings, tmp_path):
        """Create episodic memory with temp directory."""
        mem = EpisodicMemory(temp_settings)
        mem._episodes_dir = tmp_path / "episodes"
        mem._episodes_dir.mkdir(parents=True, exist_ok=True)
        return mem

    def test_record_episode(self, memory):
        """Test recording a research episode."""
        episode = ResearchEpisode(
            session_id="test-session-001",
            query="Test query",
            query_type="analytical",
            tools_used={"web_search": {"success_rate": 0.8, "calls": 3}},
            duration_seconds=120.0,
        )
        memory.record_episode(episode)
        assert memory.episode_count == 1

    def test_get_best_strategy_no_data(self, memory):
        """Test strategy recommendation with no past data."""
        rec = memory.get_best_strategy("risk_assessment")
        assert rec.based_on_episodes == 0
        assert "No past episodes" in rec.notes

    def test_get_best_strategy_with_data(self, memory):
        """Test strategy recommendation with past episodes."""
        for i in range(3):
            episode = ResearchEpisode(
                session_id=f"session-{i}",
                query=f"Risk query {i}",
                query_type="risk_assessment",
                tools_used={
                    "sec_filing_search": {"success_rate": 0.9},
                    "web_search": {"success_rate": 0.7},
                },
                duration_seconds=100.0,
            )
            memory.record_episode(episode)

        rec = memory.get_best_strategy("risk_assessment")
        assert rec.based_on_episodes == 3
        assert len(rec.recommended_tools) > 0

    def test_get_error_patterns(self, memory):
        """Test error pattern identification."""
        episode = ResearchEpisode(
            session_id="error-session",
            query="Error query",
            query_type="analytical",
            errors=[
                {"category": "tool_execution", "message": "API timeout"},
                {"category": "tool_execution", "message": "Rate limited"},
            ],
        )
        memory.record_episode(episode)

        patterns = memory.get_error_patterns()
        assert "tool_execution" in patterns
        assert len(patterns["tool_execution"]) == 2
