"""
Query Analyzer — Classifies and decomposes incoming research queries.

Analyzes queries for type, complexity, entities, ambiguity, and temporal context.
Generates sub-queries for multi-source retrieval and identifies edge cases.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class AnalyzedQuery:
    """Result of query analysis.

    Attributes:
        original_query: The raw user query.
        query_type: Classified type (factual, analytical, comparative, risk_assessment, sector_analysis).
        complexity: Assessed complexity (simple, moderate, complex).
        entities: Identified companies/sectors/topics with tickers.
        time_period: Detected or inferred time period.
        ambiguity_level: How ambiguous the query is (low, medium, high).
        ambiguities: List of identified ambiguities.
        assumptions: Assumptions made for disambiguation.
        edge_cases: Detected edge cases (private company, new IPO, etc.).
        sub_queries: Decomposed retrieval queries for multi-source search.
        recommended_tools: Tools suggested for this query type.
        estimated_tool_calls: Estimated number of tool calls needed.
    """

    original_query: str = ""
    query_type: str = "analytical"
    complexity: str = "moderate"
    entities: list[dict[str, str]] = field(default_factory=list)
    time_period: str = ""
    ambiguity_level: str = "low"
    ambiguities: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    recommended_tools: list[str] = field(default_factory=list)
    estimated_tool_calls: int = 10


# ── Common Ticker Patterns ────────────────────────────────────────────────────

WELL_KNOWN_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman sachs": "GS", "morgan stanley": "MS", "bank of america": "BAC",
    "wells fargo": "WFC", "berkshire hathaway": "BRK-B", "johnson & johnson": "JNJ",
    "procter & gamble": "PG", "coca-cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "walmart": "WMT", "disney": "DIS", "intel": "INTC", "amd": "AMD",
    "salesforce": "CRM", "adobe": "ADBE", "paypal": "PYPL", "visa": "V",
    "mastercard": "MA", "boeing": "BA", "ibm": "IBM", "oracle": "ORCL",
    "cisco": "CSCO", "qualcomm": "QCOM", "uber": "UBER", "airbnb": "ABNB",
    "spotify": "SPOT", "snowflake": "SNOW", "palantir": "PLTR",
}


class QueryAnalyzer:
    """Analyzes incoming financial research queries.

    Performs:
    1. Entity extraction (company names → tickers)
    2. Query type classification
    3. Complexity assessment
    4. Ambiguity detection
    5. Sub-query decomposition for retrieval
    6. Edge case identification
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def analyze(self, query: str) -> AnalyzedQuery:
        """Analyze a research query and return structured analysis.

        This is the rule-based fast path. For complex queries, the LLM-based
        disambiguation (in disambiguation.py) provides deeper analysis.

        Args:
            query: Raw user query string.

        Returns:
            AnalyzedQuery with all fields populated.
        """
        result = AnalyzedQuery(original_query=query)

        # Step 1: Extract entities (companies, tickers)
        result.entities = self._extract_entities(query)

        # Step 2: Classify query type
        result.query_type = self._classify_type(query)

        # Step 3: Assess complexity
        result.complexity = self._assess_complexity(query, result.entities)

        # Step 4: Detect time period
        result.time_period = self._detect_time_period(query)

        # Step 5: Check ambiguity
        result.ambiguity_level, result.ambiguities = self._check_ambiguity(
            query, result.entities
        )

        # Step 6: Generate sub-queries
        result.sub_queries = self._generate_sub_queries(
            query, result.entities, result.query_type
        )

        # Step 7: Recommend tools
        result.recommended_tools = self._recommend_tools(result.query_type)

        # Step 8: Estimate tool calls
        result.estimated_tool_calls = self._estimate_tool_calls(
            result.complexity, len(result.entities)
        )

        # Step 9: Edge cases
        result.edge_cases = self._detect_edge_cases(query, result.entities)

        return result

    def _extract_entities(self, query: str) -> list[dict[str, str]]:
        """Extract company names and ticker symbols from the query."""
        entities = []
        query_lower = query.lower()

        # Check for explicit ticker symbols (uppercase 1-5 letter words)
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, query)
        common_words = {
            "A", "I", "Q", "AND", "OR", "THE", "FOR", "INC", "CEO", "CFO",
            "CTO", "IPO", "SEC", "DCF", "YOY", "QOQ", "ROE", "ROA", "EPS",
            "PE", "PB", "FCF", "GDP", "ETF", "US", "UK", "EU", "AI",
        }
        for ticker in potential_tickers:
            if ticker not in common_words and len(ticker) >= 2:
                entities.append({"name": ticker, "ticker": ticker, "type": "company"})

        # Check for known company names
        for name, ticker in WELL_KNOWN_TICKERS.items():
            if name in query_lower:
                # Avoid duplicates
                if not any(e["ticker"] == ticker for e in entities):
                    entities.append({"name": name.title(), "ticker": ticker, "type": "company"})

        # Check for sector/industry keywords
        sectors = {
            "technology": "Technology", "tech": "Technology",
            "healthcare": "Healthcare", "pharma": "Healthcare",
            "financial": "Financials", "banking": "Financials", "banks": "Financials",
            "energy": "Energy", "oil": "Energy",
            "consumer": "Consumer", "retail": "Consumer",
            "industrial": "Industrials", "manufacturing": "Industrials",
            "real estate": "Real Estate", "reit": "Real Estate",
        }
        for keyword, sector in sectors.items():
            if keyword in query_lower:
                entities.append({"name": sector, "ticker": "", "type": "sector"})
                break

        return entities

    def _classify_type(self, query: str) -> str:
        """Classify the query type based on keywords and structure."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["compare", "versus", "vs", "comparison", "between"]):
            return "comparative"
        if any(w in query_lower for w in ["risk", "threat", "danger", "vulnerability"]):
            return "risk_assessment"
        if any(w in query_lower for w in ["sector", "industry", "market overview"]):
            return "sector_analysis"
        if any(w in query_lower for w in [
            "what was", "how much", "what is the", "revenue", "earnings",
            "price", "market cap"
        ]):
            return "factual"
        return "analytical"

    def _assess_complexity(
        self, query: str, entities: list[dict[str, str]]
    ) -> str:
        """Assess query complexity based on entity count and scope."""
        company_count = sum(1 for e in entities if e["type"] == "company")
        query_lower = query.lower()

        if company_count >= 3 or "sector" in query_lower:
            return "complex"
        if company_count >= 2 or any(
            w in query_lower
            for w in ["comprehensive", "detailed", "full report", "deep dive"]
        ):
            return "complex"
        if company_count == 1 and len(query.split()) > 15:
            return "moderate"
        if company_count <= 1 and len(query.split()) <= 10:
            return "simple"
        return "moderate"

    def _detect_time_period(self, query: str) -> str:
        """Detect time period references in the query."""
        # Year patterns
        year_match = re.findall(r'\b(20[12]\d)\b', query)
        if year_match:
            return f"Year(s): {', '.join(year_match)}"

        # Quarter patterns
        quarter_match = re.findall(r'(Q[1-4])\s*(20[12]\d)', query, re.IGNORECASE)
        if quarter_match:
            return ", ".join(f"{q} {y}" for q, y in quarter_match)

        # Relative time
        query_lower = query.lower()
        if "recent" in query_lower or "latest" in query_lower:
            return "Most recent available"
        if "last year" in query_lower:
            return "Previous fiscal year"
        if "past 5 years" in query_lower or "five year" in query_lower:
            return "5-year historical"

        return "Most recent available (default)"

    def _check_ambiguity(
        self, query: str, entities: list[dict[str, str]]
    ) -> tuple[str, list[str]]:
        """Detect potential ambiguities in the query."""
        ambiguities = []
        query_lower = query.lower()

        # No specific company identified
        if not any(e["type"] == "company" for e in entities):
            if not any(e["type"] == "sector" for e in entities):
                ambiguities.append("No specific company or sector identified in query")

        # Ambiguous company names
        if "amazon" in query_lower and "aws" not in query_lower:
            ambiguities.append(
                "Amazon could refer to e-commerce, AWS cloud, advertising, or logistics"
            )
        if "apple" in query_lower:
            pass  # Generally unambiguous in financial context
        if "meta" in query_lower and "metadata" not in query_lower:
            pass  # Generally refers to Meta Platforms

        # Vague scope
        if any(w in query_lower for w in ["analyze", "research", "look into"]):
            if len(query.split()) < 8:
                ambiguities.append("Query scope is broad — unclear which aspects to focus on")

        level = "low"
        if len(ambiguities) >= 2:
            level = "high"
        elif len(ambiguities) == 1:
            level = "medium"

        return level, ambiguities

    def _generate_sub_queries(
        self,
        query: str,
        entities: list[dict[str, str]],
        query_type: str,
    ) -> list[str]:
        """Decompose the query into specific retrieval sub-queries."""
        sub_queries = []
        companies = [e for e in entities if e["type"] == "company"]

        for company in companies:
            name = company["name"]
            ticker = company["ticker"]

            if query_type == "risk_assessment":
                sub_queries.extend([
                    f"{name} {ticker} risk factors 2024 2025",
                    f"{name} {ticker} SEC filing risk disclosures",
                    f"{name} {ticker} regulatory challenges",
                    f"{name} {ticker} competitive threats",
                ])
            elif query_type == "comparative":
                sub_queries.extend([
                    f"{name} {ticker} financial performance revenue growth",
                    f"{name} {ticker} market position competitive advantages",
                ])
            elif query_type == "factual":
                sub_queries.append(f"{name} {ticker} financial data latest")
            else:  # analytical
                sub_queries.extend([
                    f"{name} {ticker} financial performance revenue margins",
                    f"{name} {ticker} recent news developments 2024 2025",
                    f"{name} {ticker} competitive position market share",
                    f"{name} {ticker} growth strategy outlook",
                ])

        if not sub_queries:
            sub_queries.append(query)

        return sub_queries

    def _recommend_tools(self, query_type: str) -> list[str]:
        """Recommend tools based on query type."""
        base_tools = ["vector_db_search", "company_profile"]

        type_tools = {
            "factual": ["financial_data_api", "sec_filing_search"],
            "analytical": [
                "financial_data_api", "sec_filing_search",
                "web_search", "earnings_transcript", "news_sentiment",
            ],
            "comparative": [
                "peer_comparison", "financial_data_api",
                "web_search", "calculation_engine",
            ],
            "risk_assessment": [
                "sec_filing_search", "web_search",
                "news_sentiment", "earnings_transcript",
            ],
            "sector_analysis": [
                "web_search", "peer_comparison",
                "financial_data_api", "news_sentiment",
            ],
        }

        return base_tools + type_tools.get(query_type, [])

    def _estimate_tool_calls(self, complexity: str, entity_count: int) -> int:
        """Estimate the number of tool calls needed."""
        base = {"simple": 4, "moderate": 8, "complex": 15}.get(complexity, 8)
        return min(base + (entity_count * 3), 20)

    def _detect_edge_cases(
        self, query: str, entities: list[dict[str, str]]
    ) -> list[str]:
        """Detect potential edge cases."""
        edge_cases = []
        query_lower = query.lower()

        if any(w in query_lower for w in ["private company", "privately held", "pre-ipo"]):
            edge_cases.append("Private company — no SEC filings or public financial data available")
        if any(w in query_lower for w in ["ipo", "newly listed", "just listed"]):
            edge_cases.append("Newly IPO'd company — limited historical financial data")
        if any(w in query_lower for w in ["acquisition", "merger", "being acquired"]):
            edge_cases.append("M&A situation — rapidly changing circumstances, temporal sensitivity")
        if any(w in query_lower for w in ["bankruptcy", "chapter 11", "restructuring"]):
            edge_cases.append("Distressed company — financial data may not reflect going-concern status")

        return edge_cases
