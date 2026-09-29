"""
Disambiguation Module — Handles ambiguous queries using LLM-powered analysis.

When the rule-based QueryAnalyzer detects medium/high ambiguity, this module
uses the LLM to generate clarifying questions or make documented assumptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class DisambiguationResult:
    """Result of disambiguation analysis.

    Attributes:
        needs_clarification: Whether the query requires user input.
        clarifying_questions: Questions to ask the user (if any).
        assumptions: Documented assumptions made for proceeding.
        interpreted_query: The disambiguated interpretation of the query.
        confidence: Confidence in the interpretation (0-1).
        edge_case_handling: How edge cases will be handled.
    """

    needs_clarification: bool = False
    clarifying_questions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    interpreted_query: str = ""
    confidence: float = 1.0
    edge_case_handling: dict[str, str] = field(default_factory=dict)


class QueryDisambiguator:
    """Resolves ambiguous queries with documented assumptions.

    For autonomous operation, the disambiguator makes reasonable assumptions
    rather than blocking on user input. All assumptions are documented
    in the final report's methodology section.

    Handles edge cases:
    - Private companies (no SEC filings)
    - Newly IPO'd companies (limited history)
    - Ambiguous company names (multiple interpretations)
    - Missing time period specifications
    - Overly broad scope
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def disambiguate(
        self,
        query: str,
        ambiguity_level: str,
        ambiguities: list[str],
        entities: list[dict[str, str]],
    ) -> DisambiguationResult:
        """Disambiguate a query using rule-based heuristics.

        For the autonomous agent, we make documented assumptions rather than
        asking the user for clarification. Each assumption is logged for
        inclusion in the methodology section.

        Args:
            query: Original user query.
            ambiguity_level: From QueryAnalyzer ("low", "medium", "high").
            ambiguities: List of identified ambiguities.
            entities: Extracted entities from QueryAnalyzer.

        Returns:
            DisambiguationResult with assumptions and interpreted query.
        """
        result = DisambiguationResult(interpreted_query=query)

        if ambiguity_level == "low":
            result.confidence = 0.95
            return result

        # Handle each identified ambiguity
        for ambiguity in ambiguities:
            assumption = self._resolve_ambiguity(ambiguity, query, entities)
            if assumption:
                result.assumptions.append(assumption)

        # Handle edge cases
        result.edge_case_handling = self._handle_edge_cases(query, entities)

        # Set confidence based on number of assumptions
        if len(result.assumptions) == 0:
            result.confidence = 0.9
        elif len(result.assumptions) <= 2:
            result.confidence = 0.75
        else:
            result.confidence = 0.6

        # For very high ambiguity with no entities, flag for clarification
        if ambiguity_level == "high" and not any(
            e["type"] == "company" for e in entities
        ):
            result.needs_clarification = True
            result.clarifying_questions = [
                "Which specific company or companies should be analyzed?",
                "What time period should the analysis cover?",
                "What aspects are most important (financial performance, risks, competitive position)?",
            ]

        return result

    def _resolve_ambiguity(
        self,
        ambiguity: str,
        query: str,
        entities: list[dict[str, str]],
    ) -> str:
        """Generate a documented assumption to resolve an ambiguity."""
        ambiguity_lower = ambiguity.lower()

        if "no specific company" in ambiguity_lower:
            return (
                "No specific company identified. Interpreting as a general market "
                "or topic analysis based on available context."
            )

        if "amazon" in ambiguity_lower:
            return (
                "Assuming 'Amazon' refers to Amazon.com Inc. (AMZN) as a whole, "
                "covering e-commerce, AWS, advertising, and logistics segments."
            )

        if "scope is broad" in ambiguity_lower:
            return (
                "Query scope is broad. Defaulting to a comprehensive analysis covering "
                "financial performance, risk factors, competitive positioning, and outlook."
            )

        return f"Assumption made regarding: {ambiguity}"

    def _handle_edge_cases(
        self, query: str, entities: list[dict[str, str]]
    ) -> dict[str, str]:
        """Determine handling strategies for detected edge cases."""
        handling = {}
        query_lower = query.lower()

        if "private" in query_lower:
            handling["private_company"] = (
                "Private company detected. SEC filings and public financial data "
                "will not be available. Relying on news, web search, and any "
                "available industry reports."
            )

        if "ipo" in query_lower or "newly listed" in query_lower:
            handling["new_ipo"] = (
                "Newly IPO'd company detected. Historical financial data may be "
                "limited. Will supplement with S-1/prospectus data and industry "
                "comparisons."
            )

        if "acquisition" in query_lower or "merger" in query_lower:
            handling["ma_activity"] = (
                "M&A activity detected. Data may be rapidly changing. Will flag "
                "temporal sensitivity and note that financial projections may not "
                "account for post-merger integration."
            )

        return handling
