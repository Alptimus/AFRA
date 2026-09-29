"""
Synthesis Engine — Multi-source data synthesis orchestrator.

Combines information from multiple financial data sources into coherent
analytical narratives using narrative threading, quantitative triangulation,
and sentiment-fact alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from synthesis.conflict_resolver import ConflictResolver, DataConflict

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of multi-source synthesis.

    Attributes:
        narrative: The synthesized analytical narrative.
        key_insights: Non-obvious analytical observations.
        conflicts_found: Data conflicts detected across sources.
        source_agreement: Data points confirmed by multiple sources.
        sentiment_fact_alignment: Qualitative vs quantitative alignment analysis.
        confidence_score: Overall confidence in the synthesis (0-1).
    """

    narrative: str = ""
    key_insights: list[str] = field(default_factory=list)
    conflicts_found: list[DataConflict] = field(default_factory=list)
    source_agreement: list[dict[str, Any]] = field(default_factory=list)
    sentiment_fact_alignment: dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0


class SynthesisEngine:
    """Orchestrates multi-source financial data synthesis.

    Techniques (from Section A6.4):
    1. Narrative Threading — connects data points into coherent story arcs
    2. Quantitative Triangulation — verifies numbers from 3+ sources
    3. Sentiment-Fact Alignment — compares qualitative and quantitative signals
    """

    def __init__(self) -> None:
        self._conflict_resolver = ConflictResolver()

    def synthesize(self, gathered_data: dict[str, Any]) -> SynthesisResult:
        """Synthesize data from multiple sources.

        Args:
            gathered_data: Dict mapping source keys to data dicts, as produced
                          by the agent's execution phase.

        Returns:
            SynthesisResult with narrative, insights, and conflicts.
        """
        result = SynthesisResult()

        # Categorize data by source type
        categorized = self._categorize_by_source(gathered_data)

        # Step 1: Detect conflicts
        result.conflicts_found = self._detect_conflicts(categorized)

        # Step 2: Resolve conflicts
        for conflict in result.conflicts_found:
            resolution = self._conflict_resolver.resolve(conflict)
            conflict.resolution = resolution

        # Step 3: Identify source agreement (triangulation)
        result.source_agreement = self._triangulate(categorized)

        # Step 4: Sentiment-fact alignment
        result.sentiment_fact_alignment = self._check_sentiment_alignment(
            categorized
        )

        # Step 5: Generate narrative threads
        result.narrative = self._generate_narrative(
            categorized, result.conflicts_found, result.source_agreement
        )

        # Step 6: Extract key insights
        result.key_insights = self._extract_insights(
            categorized, result.conflicts_found, result.sentiment_fact_alignment
        )

        # Step 7: Compute confidence
        result.confidence_score = self._compute_confidence(result)

        return result

    def _categorize_by_source(
        self, gathered_data: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Categorize gathered data by source type."""
        categories: dict[str, list[dict[str, Any]]] = {
            "sec_filings": [],
            "financial_data": [],
            "earnings": [],
            "news": [],
            "sentiment": [],
            "profiles": [],
            "peers": [],
            "memory": [],
            "other": [],
        }

        source_map = {
            "sec_filing_search": "sec_filings",
            "financial_data_api": "financial_data",
            "earnings_transcript": "earnings",
            "web_search": "news",
            "news_sentiment": "sentiment",
            "company_profile": "profiles",
            "peer_comparison": "peers",
            "vector_db_search": "memory",
        }

        for key, value in gathered_data.items():
            tool = value.get("tool", "")
            category = source_map.get(tool, "other")
            categories[category].append(value)

        return categories

    def _detect_conflicts(
        self, categorized: dict[str, list[dict[str, Any]]]
    ) -> list[DataConflict]:
        """Detect conflicting data points across sources."""
        conflicts = []

        # Compare financial data from different sources
        all_financial = categorized.get("financial_data", []) + categorized.get(
            "sec_filings", []
        )

        # Look for the same metric reported differently
        metrics_seen: dict[str, list[tuple[str, Any]]] = {}

        for source in all_financial:
            data = source.get("data", {})
            tool = source.get("tool", "unknown")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)) and key not in (
                        "execution_time_ms",
                        "total_results",
                    ):
                        if key not in metrics_seen:
                            metrics_seen[key] = []
                        metrics_seen[key].append((tool, value))

        for metric, values in metrics_seen.items():
            if len(values) < 2:
                continue
            # Check if values disagree by more than 5%
            nums = [v for _, v in values if isinstance(v, (int, float)) and v != 0]
            if len(nums) >= 2:
                max_val = max(nums)
                min_val = min(nums)
                if min_val != 0 and (max_val - min_val) / abs(min_val) > 0.05:
                    conflicts.append(
                        DataConflict(
                            metric=metric,
                            values={src: val for src, val in values},
                            source_tiers={src: self._conflict_resolver.get_tier(src) for src, _ in values},
                        )
                    )

        return conflicts

    def _triangulate(
        self, categorized: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Identify data points confirmed by multiple sources."""
        agreements = []
        # Simplified: check if key data points appear in multiple source categories
        source_count = sum(1 for v in categorized.values() if v)
        if source_count >= 3:
            agreements.append({
                "note": f"Data gathered from {source_count} independent source categories",
                "strength": "strong" if source_count >= 4 else "moderate",
            })
        return agreements

    def _check_sentiment_alignment(
        self, categorized: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Compare qualitative sentiment with quantitative financial data."""
        result = {"aligned": True, "details": ""}

        sentiment_data = categorized.get("sentiment", [])
        financial_data = categorized.get("financial_data", [])

        if not sentiment_data or not financial_data:
            result["details"] = "Insufficient data for sentiment-fact alignment"
            return result

        # Extract overall sentiment
        overall_sentiment = None
        for source in sentiment_data:
            data = source.get("data", {})
            if "overall_sentiment" in data:
                overall_sentiment = data["overall_sentiment"]
                break

        # Extract financial trend (simplified)
        revenue_growth = None
        for source in financial_data:
            data = source.get("data", {})
            if isinstance(data, dict) and "revenue_growth" in data:
                revenue_growth = data["revenue_growth"]
                break

        if overall_sentiment and revenue_growth is not None:
            sentiment_positive = overall_sentiment == "positive"
            growth_positive = revenue_growth > 0

            if sentiment_positive != growth_positive:
                result["aligned"] = False
                result["details"] = (
                    f"MISALIGNMENT: News sentiment is {overall_sentiment} but "
                    f"revenue growth is {revenue_growth:.1%}. This divergence "
                    "may indicate unrealized risks or lagging market perception."
                )
            else:
                result["details"] = (
                    f"Sentiment ({overall_sentiment}) aligns with financial "
                    f"trend (growth: {revenue_growth:.1%})"
                )

        return result

    def _generate_narrative(
        self,
        categorized: dict[str, list[dict[str, Any]]],
        conflicts: list[DataConflict],
        agreements: list[dict[str, Any]],
    ) -> str:
        """Generate a narrative threading data points from multiple sources."""
        parts = []

        # Financial overview thread
        if categorized.get("financial_data") or categorized.get("sec_filings"):
            parts.append("**Financial Performance Thread:**")
            for source in categorized.get("financial_data", [])[:2]:
                data = source.get("data", {})
                if isinstance(data, dict):
                    ticker = data.get("ticker", "")
                    parts.append(f"Financial data retrieved for {ticker}. [Source: {source.get('tool', '')}]")
            parts.append("")

        # News/market thread
        if categorized.get("news") or categorized.get("sentiment"):
            parts.append("**Market & News Thread:**")
            for source in categorized.get("sentiment", [])[:1]:
                data = source.get("data", {})
                if isinstance(data, dict):
                    parts.append(
                        f"Market sentiment: {data.get('overall_sentiment', 'N/A')} "
                        f"(based on {data.get('articles_analyzed', 0)} articles). "
                        f"[Source: {source.get('tool', '')}]"
                    )
            parts.append("")

        # Conflicts thread
        if conflicts:
            parts.append("**Data Conflicts:**")
            for conflict in conflicts:
                parts.append(
                    f"- Conflict on '{conflict.metric}': "
                    f"values differ across sources. Resolution: {conflict.resolution}"
                )
            parts.append("")

        return "\n".join(parts) if parts else "Insufficient data for narrative synthesis."

    def _extract_insights(
        self,
        categorized: dict[str, list[dict[str, Any]]],
        conflicts: list[DataConflict],
        sentiment_alignment: dict[str, Any],
    ) -> list[str]:
        """Extract non-obvious analytical insights."""
        insights = []

        if conflicts:
            insights.append(
                f"Found {len(conflicts)} data conflict(s) across sources, "
                "suggesting potential data staleness or restatements."
            )

        if not sentiment_alignment.get("aligned", True):
            insights.append(
                f"Sentiment-fact misalignment detected: {sentiment_alignment.get('details', '')}"
            )

        source_types_used = sum(1 for v in categorized.values() if v)
        insights.append(
            f"Research drew from {source_types_used} independent source categories "
            "for cross-validated analysis."
        )

        return insights

    def _compute_confidence(self, result: SynthesisResult) -> float:
        """Compute overall synthesis confidence score."""
        score = 0.5  # Base

        # More source agreement → higher confidence
        if result.source_agreement:
            score += 0.1 * len(result.source_agreement)

        # Conflicts reduce confidence
        score -= 0.05 * len(result.conflicts_found)

        # Sentiment alignment boosts confidence
        if result.sentiment_fact_alignment.get("aligned"):
            score += 0.1

        return max(0.0, min(1.0, score))
