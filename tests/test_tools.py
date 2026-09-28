"""
Test Suite — Tool Registry and individual tool tests.

Tests cover:
- Tool registry registration, discovery, and execution
- Input validation against JSON schemas
- Fallback chain execution
- Rate limiting and caching
- Individual tool implementations (with mocks)
"""

from __future__ import annotations

import asyncio
import pytest

from config.settings import Settings
from tools.base import BaseTool, ToolResult, ValidationError
from tools.tool_registry import ToolRegistry, RateLimiter
from tools.calculator import CalculationEngineTool
from tools.report_gen import ReportGeneratorTool
from tools.fact_checker import FactCheckerTool


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    """Create a test settings instance with no API keys."""
    return Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        llm_model="gpt-4o-mini",
        vector_db="chroma",
        chroma_persist_dir="./data/test_chroma",
    )


@pytest.fixture
def registry(settings):
    """Create a test registry."""
    return ToolRegistry(settings)


class MockTool(BaseTool):
    """A mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing purposes"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Test query"},
            "count": {"type": "integer", "description": "Count"},
        },
        "required": ["query"],
    }
    fallback_tools = ["mock_fallback"]

    async def _execute(self, **kwargs):
        return ToolResult(
            success=True,
            data={"echo": kwargs.get("query", ""), "count": kwargs.get("count", 0)},
            source=self.name,
        )


class MockFailingTool(BaseTool):
    """A tool that always fails, for testing fallback chains."""

    name = "mock_failing_tool"
    description = "Always fails"
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    fallback_tools = ["mock_tool"]

    async def _execute(self, **kwargs):
        return ToolResult(success=False, error="Simulated failure", source=self.name)


class MockFallbackTool(BaseTool):
    """Fallback tool for testing."""

    name = "mock_fallback"
    description = "Fallback tool"
    parameters_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}

    async def _execute(self, **kwargs):
        return ToolResult(
            success=True, data={"fallback": True}, source=self.name
        )


# ── Registry Tests ────────────────────────────────────────────────────────────


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_tool(self, registry):
        """Test basic tool registration."""
        tool = MockTool()
        registry.register(tool)
        assert "mock_tool" in registry.list_tools()
        assert registry.get_tool("mock_tool") is tool

    def test_register_duplicate_raises(self, registry):
        """Test that registering a duplicate tool raises ValueError."""
        tool = MockTool()
        registry.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_list_tools(self, registry):
        """Test listing registered tools."""
        registry.register(MockTool())
        registry.register(MockFallbackTool())
        tools = registry.list_tools()
        assert "mock_tool" in tools
        assert "mock_fallback" in tools
        assert len(tools) == 2

    def test_get_tool_schemas_openai(self, registry):
        """Test schema export in OpenAI format."""
        registry.register(MockTool())
        schemas = registry.get_tool_schemas(format="openai")
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mock_tool"

    def test_get_tool_schemas_anthropic(self, registry):
        """Test schema export in Anthropic format."""
        registry.register(MockTool())
        schemas = registry.get_tool_schemas(format="anthropic")
        assert len(schemas) == 1
        assert schemas[0]["name"] == "mock_tool"
        assert "input_schema" in schemas[0]

    @pytest.mark.asyncio
    async def test_execute_tool(self, registry):
        """Test basic tool execution through the registry."""
        registry.register(MockTool())
        result = await registry.execute("mock_tool", query="hello")
        assert result.success is True
        assert result.data["echo"] == "hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry):
        """Test executing a non-existent tool returns error."""
        result = await registry.execute("nonexistent_tool")
        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, registry):
        """Test fallback chain execution when primary tool fails."""
        registry.register(MockFailingTool())
        registry.register(MockTool())
        registry.register(MockFallbackTool())

        result = await registry.execute_with_fallback("mock_failing_tool")
        # Should fallback to mock_tool or mock_fallback
        assert result.success is True

    @pytest.mark.asyncio
    async def test_caching(self, registry):
        """Test that results are cached."""
        registry.register(MockTool())

        # First call
        result1 = await registry.execute("mock_tool", query="cached_test")
        # Second call should hit cache
        result2 = await registry.execute("mock_tool", query="cached_test")

        assert result1.success is True
        assert result2.success is True
        # Only one call should be logged (second was cached)
        assert registry.total_calls == 1

    def test_tool_stats(self, registry):
        """Test tool statistics tracking."""
        registry.register(MockTool())
        asyncio.get_event_loop().run_until_complete(
            registry.execute("mock_tool", query="test", use_cache=False)
        )
        stats = registry.get_tool_stats()
        assert "mock_tool" in stats
        assert stats["mock_tool"]["call_count"] == 1
        assert stats["mock_tool"]["success_rate"] == 1.0


# ── Validation Tests ──────────────────────────────────────────────────────────


class TestToolValidation:
    """Tests for tool input validation."""

    def test_valid_inputs(self):
        """Test that valid inputs pass validation."""
        tool = MockTool()
        tool.validate_inputs(query="hello", count=5)  # Should not raise

    def test_missing_required_param(self):
        """Test that missing required params raise ValidationError."""
        tool = MockTool()
        with pytest.raises(ValidationError, match="Missing required parameter"):
            tool.validate_inputs(count=5)  # Missing 'query'

    def test_wrong_type(self):
        """Test that wrong types raise ValidationError."""
        tool = MockTool()
        with pytest.raises(ValidationError, match="expects string"):
            tool.validate_inputs(query=123)  # Should be string


# ── Calculator Tool Tests ─────────────────────────────────────────────────────


class TestCalculatorTool:
    """Tests for the CalculationEngineTool."""

    @pytest.fixture
    def calc(self):
        return CalculationEngineTool()

    @pytest.mark.asyncio
    async def test_ratio_calculation(self, calc):
        """Test basic ratio calculation."""
        result = await calc.execute(
            calculation_type="ratio",
            inputs={"numerator": 100, "denominator": 200, "name": "test_ratio"},
        )
        assert result.success is True
        assert result.data["result"] == 0.5

    @pytest.mark.asyncio
    async def test_cagr_calculation(self, calc):
        """Test CAGR calculation."""
        result = await calc.execute(
            calculation_type="cagr",
            inputs={"beginning_value": 100, "ending_value": 200, "periods": 5},
        )
        assert result.success is True
        assert 0.14 < result.data["result"] < 0.15  # ~14.87%

    @pytest.mark.asyncio
    async def test_growth_rate(self, calc):
        """Test growth rate calculation."""
        result = await calc.execute(
            calculation_type="growth_rate",
            inputs={"old_value": 100, "new_value": 115, "name": "revenue_growth"},
        )
        assert result.success is True
        assert result.data["result"] == 0.15

    @pytest.mark.asyncio
    async def test_dcf_calculation(self, calc):
        """Test DCF valuation."""
        result = await calc.execute(
            calculation_type="dcf",
            inputs={
                "cash_flows": [100, 110, 121, 133, 146],
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.03,
            },
        )
        assert result.success is True
        assert result.data["enterprise_value"] > 0

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calc):
        """Test handling of division by zero."""
        result = await calc.execute(
            calculation_type="ratio",
            inputs={"numerator": 100, "denominator": 0, "name": "bad_ratio"},
        )
        assert result.success is True
        assert result.data["result"] is None


# ── Report Generator Tests ────────────────────────────────────────────────────


class TestReportGenerator:
    """Tests for the ReportGeneratorTool."""

    @pytest.fixture
    def gen(self):
        return ReportGeneratorTool()

    @pytest.mark.asyncio
    async def test_generate_report(self, gen):
        """Test basic report generation."""
        result = await gen.execute(
            template="company_profile",
            sections={
                "Company Overview": "Apple Inc. is a technology company.",
                "Business Description": "Apple designs and sells electronics.",
            },
            sources=["company_profile", "web_search"],
        )
        assert result.success is True
        assert "Apple Inc." in result.data["report"]
        assert result.data["sections_completed"] >= 2

    @pytest.mark.asyncio
    async def test_unknown_template(self, gen):
        """Test that unknown templates return an error."""
        result = await gen.execute(
            template="nonexistent",
            sections={},
        )
        assert result.success is False
        assert "Unknown template" in result.error


# ── Rate Limiter Tests ────────────────────────────────────────────────────────


class TestRateLimiter:
    """Tests for the RateLimiter."""

    def test_allows_within_limit(self):
        limiter = RateLimiter(max_calls=3, window_seconds=60.0)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_calls=2, window_seconds=60.0)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False
