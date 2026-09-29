"""
Tests for the agent core — query analyzer, disambiguation, error handler,
circuit breaker, and response parser.
"""

from __future__ import annotations

import asyncio
import pytest

from agent.query_analyzer import QueryAnalyzer, AnalyzedQuery
from agent.disambiguation import QueryDisambiguator, DisambiguationResult
from agent.error_handler import (
    AgentError, ErrorCategory, ErrorHandler, ErrorSeverity,
)
from agent.circuit_breaker import CircuitBreaker, CircuitState
from agent.parser import ResponseParser, ParsedResponse, ParsedToolCall
from agent.fallback_chains import get_fallback_chain, get_all_chains


# ── Query Analyzer Tests ──────────────────────────────────────────────────────


class TestQueryAnalyzer:
    """Tests for the QueryAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return QueryAnalyzer()

    def test_extract_ticker(self, analyzer):
        """Test ticker extraction from query."""
        result = analyzer.analyze("Analyze AAPL financial performance")
        tickers = [e["ticker"] for e in result.entities if e["type"] == "company"]
        assert "AAPL" in tickers

    def test_extract_company_name(self, analyzer):
        """Test company name extraction."""
        result = analyzer.analyze("What is Tesla's revenue growth?")
        tickers = [e["ticker"] for e in result.entities if e["type"] == "company"]
        assert "TSLA" in tickers

    def test_classify_risk_assessment(self, analyzer):
        """Test risk query classification."""
        result = analyzer.analyze("Prepare a risk assessment for Microsoft")
        assert result.query_type == "risk_assessment"

    def test_classify_comparative(self, analyzer):
        """Test comparative query classification."""
        result = analyzer.analyze("Compare JPMorgan versus Goldman Sachs")
        assert result.query_type == "comparative"

    def test_classify_factual(self, analyzer):
        """Test factual query classification."""
        result = analyzer.analyze("What was Apple's revenue in 2024?")
        assert result.query_type == "factual"

    def test_complexity_simple(self, analyzer):
        """Test simple complexity assessment."""
        result = analyzer.analyze("AAPL revenue")
        assert result.complexity == "simple"

    def test_complexity_complex(self, analyzer):
        """Test complex query assessment."""
        result = analyzer.analyze(
            "Prepare a comprehensive detailed analysis comparing Apple, Microsoft, "
            "and Google across financial performance, risk factors, and competitive position"
        )
        assert result.complexity == "complex"

    def test_time_period_detection(self, analyzer):
        """Test time period extraction."""
        result = analyzer.analyze("Tesla financial results for Q3 2024")
        assert "2024" in result.time_period

    def test_sub_query_generation(self, analyzer):
        """Test sub-query decomposition."""
        result = analyzer.analyze("Analyze Tesla's financial performance")
        assert len(result.sub_queries) >= 1

    def test_tool_recommendation(self, analyzer):
        """Test tool recommendations for query types."""
        result = analyzer.analyze("What is Tesla's risk profile?")
        assert "sec_filing_search" in result.recommended_tools

    def test_edge_case_private_company(self, analyzer):
        """Test edge case detection for private companies."""
        result = analyzer.analyze("Analyze the private company SpaceX pre-ipo")
        assert any("Private" in ec or "private" in ec.lower() for ec in result.edge_cases)


# ── Disambiguation Tests ──────────────────────────────────────────────────────


class TestQueryDisambiguator:
    """Tests for the QueryDisambiguator."""

    @pytest.fixture
    def disambiguator(self):
        return QueryDisambiguator()

    def test_low_ambiguity_pass_through(self, disambiguator):
        """Test that low-ambiguity queries pass through unchanged."""
        result = disambiguator.disambiguate(
            "Analyze AAPL", "low", [], [{"name": "Apple", "ticker": "AAPL", "type": "company"}]
        )
        assert result.confidence >= 0.9
        assert not result.needs_clarification

    def test_high_ambiguity_no_entity(self, disambiguator):
        """Test high-ambiguity query with no entities flags clarification."""
        result = disambiguator.disambiguate(
            "Analyze the market", "high", ["No specific company"], []
        )
        assert result.needs_clarification
        assert len(result.clarifying_questions) > 0

    def test_documented_assumptions(self, disambiguator):
        """Test that assumptions are documented."""
        result = disambiguator.disambiguate(
            "What about Amazon?", "medium",
            ["Amazon could refer to e-commerce, AWS, advertising, or logistics"],
            [{"name": "Amazon", "ticker": "AMZN", "type": "company"}],
        )
        assert len(result.assumptions) > 0


# ── Error Handler Tests ───────────────────────────────────────────────────────


class TestErrorHandler:
    """Tests for the ErrorHandler."""

    @pytest.fixture
    def handler(self):
        return ErrorHandler(max_retries=3, base_delay=0.01)

    def test_record_error(self, handler):
        """Test error recording."""
        error = AgentError(
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            message="API timeout",
            tool_name="web_search",
        )
        handler.record_error(error)
        assert handler.error_count == 1

    def test_recovery_tracking(self, handler):
        """Test recovery tracking."""
        error = AgentError(
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            message="API timeout",
            tool_name="web_search",
        )
        handler.record_error(error)
        handler.record_recovery(error, "Used fallback tool")
        assert handler.recovery_rate == 1.0

    def test_degradation_report(self, handler):
        """Test degradation report generation."""
        error = AgentError(
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.HIGH,
            message="All tools failed",
            tool_name="sec_filing_search",
        )
        handler.record_error(error)
        report = handler.get_degradation_report()
        assert report["has_degradation"] is True
        assert "sec_filing_search" in report["affected_tools"]

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, handler):
        """Test retry succeeding on second attempt."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await handler.retry_with_backoff(flaky_func)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, handler):
        """Test retry exhaustion raises the last exception."""
        async def always_fail():
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError, match="Permanent failure"):
            await handler.retry_with_backoff(always_fail, max_retries=2)


# ── Circuit Breaker Tests ─────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for the CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

    def test_initial_state_closed(self, breaker):
        """Test that initial state is CLOSED."""
        assert breaker.can_execute("test_tool") is True

    def test_opens_after_threshold(self, breaker):
        """Test that circuit opens after consecutive failures."""
        for _ in range(3):
            breaker.record_failure("test_tool")
        assert breaker.can_execute("test_tool") is False

    def test_success_resets_count(self, breaker):
        """Test that a success resets the failure count."""
        breaker.record_failure("test_tool")
        breaker.record_failure("test_tool")
        breaker.record_success("test_tool")  # Reset
        breaker.record_failure("test_tool")
        assert breaker.can_execute("test_tool") is True  # Still CLOSED

    def test_half_open_recovery(self, breaker):
        """Test HALF_OPEN → CLOSED transition on success."""
        # Open the circuit
        for _ in range(3):
            breaker.record_failure("test_tool")

        # Wait for recovery timeout
        import time
        time.sleep(0.15)

        # Should transition to HALF_OPEN
        assert breaker.can_execute("test_tool") is True

        # Success should close it
        breaker.record_success("test_tool")
        assert breaker.can_execute("test_tool") is True

    def test_reset(self, breaker):
        """Test circuit breaker reset."""
        for _ in range(3):
            breaker.record_failure("test_tool")
        breaker.reset("test_tool")
        assert breaker.can_execute("test_tool") is True


# ── Response Parser Tests ─────────────────────────────────────────────────────


class TestResponseParser:
    """Tests for the ResponseParser."""

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_extract_thoughts(self, parser):
        """Test thought extraction from text content."""
        content = "Thought: I need to check the financial data first.\nAction: Use financial_data_api"
        thoughts = parser._extract_thoughts(content)
        assert len(thoughts) >= 1

    def test_plain_text_is_final(self, parser):
        """Test that plain text response is marked as final."""

        class MockResponse:
            content = "Here is the final analysis of Tesla."
            tool_calls = []

        result = parser.parse_openai_response(MockResponse())
        assert result.is_final is True
        assert "Tesla" in result.final_answer


# ── Fallback Chain Tests ──────────────────────────────────────────────────────


class TestFallbackChains:
    """Tests for fallback chain definitions."""

    def test_sec_filing_fallback(self):
        """Test SEC filing tool has web_search fallback."""
        chain = get_fallback_chain("sec_filing_search")
        assert "web_search" in chain

    def test_internal_tools_no_fallback(self):
        """Test internal tools have empty fallback chains."""
        assert get_fallback_chain("calculation_engine") == []
        assert get_fallback_chain("vector_db_search") == []

    def test_all_chains_defined(self):
        """Test all chains are defined."""
        chains = get_all_chains()
        assert len(chains) >= 12
