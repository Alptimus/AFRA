"""
Tests for the synthesis engine — conflict resolution, narrative threading,
and multi-source synthesis.
"""

from __future__ import annotations

import pytest

from synthesis.conflict_resolver import ConflictResolver, DataConflict
from synthesis.narrative import NarrativeThreader, NarrativeThread
from synthesis.engine import SynthesisEngine


# ── Conflict Resolver Tests ───────────────────────────────────────────────────


class TestConflictResolver:
    """Tests for the ConflictResolver."""

    @pytest.fixture
    def resolver(self):
        return ConflictResolver()

    def test_source_tier_hierarchy(self, resolver):
        """Test the CORRECTED source hierarchy (news > social media)."""
        # SEC filings should be highest tier (1)
        assert resolver.get_tier("sec_filing_search") == 1
        # Financial APIs should be Tier 2
        assert resolver.get_tier("financial_data_api") == 2
        # Earnings transcripts should be Tier 3
        assert resolver.get_tier("earnings_transcript") == 3
        # News outlets should be Tier 4 (CORRECTED from 5)
        assert resolver.get_tier("web_search") == 4
        # Social media should be Tier 5 (CORRECTED from 4)
        assert resolver.get_tier("social_media") == 5

    def test_resolve_by_highest_tier(self, resolver):
        """Test that conflicts resolve in favor of highest-tier source."""
        conflict = DataConflict(
            metric="revenue",
            values={"sec_filing_search": 95.4e9, "web_search": 95.0e9},
            source_tiers={"sec_filing_search": 1, "web_search": 4},
        )
        resolution = resolver.resolve(conflict)
        assert "sec_filing_search" in resolution
        assert conflict.resolved_value == 95.4e9

    def test_rounding_difference_detection(self, resolver):
        """Test detection of <1% rounding differences."""
        conflict = DataConflict(
            metric="market_cap",
            values={"financial_data_api": 3.1e12, "web_search": 3.105e12},
            source_tiers={"financial_data_api": 2, "web_search": 4},
        )
        resolution = resolver.resolve(conflict)
        assert "rounding" in resolution.lower() or "highest-tier" in resolution.lower()

    def test_temporal_conflict_resolution(self, resolver):
        """Test that temporal differences are identified."""
        conflict = DataConflict(
            metric="revenue",
            values={"source_a": 95.0e9, "source_b": 90.0e9},
            source_tiers={"source_a": 2, "source_b": 2},
            temporal_info={"source_a": "Q4 2024", "source_b": "Q3 2024"},
        )
        resolution = resolver.resolve(conflict)
        assert "temporal" in resolution.lower()


# ── Narrative Threader Tests ──────────────────────────────────────────────────


class TestNarrativeThreader:
    """Tests for the NarrativeThreader."""

    @pytest.fixture
    def threader(self):
        return NarrativeThreader()

    def test_build_threads_with_financial_data(self, threader):
        """Test thread building from financial data."""
        data = {
            "financial_data": [
                {"tool": "financial_data_api", "data": {"revenue": 95.4e9, "revenueGrowth": 0.15}},
            ],
            "sentiment": [
                {"tool": "news_sentiment", "data": {"overall_sentiment": "positive", "articles_analyzed": 10}},
            ],
            "sec_filings": [],
            "earnings": [],
            "news": [],
            "profiles": [],
            "peers": [],
            "memory": [],
            "other": [],
        }
        threads = threader.build_threads(data)
        assert len(threads) >= 1

        themes = [t.theme for t in threads]
        assert "growth_trajectory" in themes

    def test_build_threads_empty_data(self, threader):
        """Test thread building with empty data."""
        data = {k: [] for k in [
            "financial_data", "sentiment", "sec_filings", "earnings",
            "news", "profiles", "peers", "memory", "other",
        ]}
        threads = threader.build_threads(data)
        assert len(threads) == 0

    def test_format_threads(self, threader):
        """Test thread formatting."""
        threads = [
            NarrativeThread(
                theme="growth_trajectory",
                thesis="Revenue growing steadily",
                data_points=[{"metric": "revenue", "value": 95e9, "source": "fmp"}],
                confidence=0.8,
            ),
        ]
        formatted = threader.format_threads(threads)
        assert "Growth Trajectory" in formatted
        assert "80%" in formatted


# ── Synthesis Engine Tests ────────────────────────────────────────────────────


class TestSynthesisEngine:
    """Tests for the SynthesisEngine."""

    @pytest.fixture
    def engine(self):
        return SynthesisEngine()

    def test_synthesize_basic(self, engine):
        """Test basic synthesis with minimal data."""
        data = {
            "step_1_company_profile": {
                "tool": "company_profile",
                "data": {"ticker": "AAPL", "company_name": "Apple Inc.", "sector": "Technology"},
                "metadata": {},
                "execution_time_ms": 200,
            },
            "step_2_financial_data_api": {
                "tool": "financial_data_api",
                "data": {"ticker": "AAPL", "statement_type": "income_statement", "data": []},
                "metadata": {},
                "execution_time_ms": 300,
            },
        }
        result = engine.synthesize(data)
        assert result.confidence_score > 0
        assert isinstance(result.narrative, str)

    def test_conflict_detection(self, engine):
        """Test that conflicts are detected across sources."""
        data = {
            "step_1_financial": {
                "tool": "financial_data_api",
                "data": {"revenue": 95.4e9, "ticker": "AAPL"},
                "metadata": {},
                "execution_time_ms": 200,
            },
            "step_2_sec": {
                "tool": "sec_filing_search",
                "data": {"revenue": 90.0e9, "ticker": "AAPL"},
                "metadata": {},
                "execution_time_ms": 300,
            },
        }
        result = engine.synthesize(data)
        # Revenue differs by >5%, should be flagged
        assert len(result.conflicts_found) >= 1

    def test_source_categorization(self, engine):
        """Test data categorization by source type."""
        data = {
            "step_1": {"tool": "sec_filing_search", "data": {}, "metadata": {}, "execution_time_ms": 100},
            "step_2": {"tool": "web_search", "data": {}, "metadata": {}, "execution_time_ms": 100},
            "step_3": {"tool": "news_sentiment", "data": {}, "metadata": {}, "execution_time_ms": 100},
        }
        categorized = engine._categorize_by_source(data)
        assert len(categorized["sec_filings"]) == 1
        assert len(categorized["news"]) == 1
        assert len(categorized["sentiment"]) == 1
