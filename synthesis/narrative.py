"""
Narrative Threading — Connects data points from multiple sources into story arcs.

Builds chronological or thematic narratives that weave together financial data,
management commentary, market conditions, and analyst perspectives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NarrativeThread:
    """A single narrative thread connecting related data points.

    Attributes:
        theme: The central theme of this thread.
        thesis: The thread's main analytical conclusion.
        data_points: Evidence supporting the thesis, with source attribution.
        confidence: How well-supported this thread is (0-1).
    """

    theme: str = ""
    thesis: str = ""
    data_points: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5


class NarrativeThreader:
    """Connects cross-source data points into coherent narrative threads.

    Identifies common themes across data from different sources and
    constructs analytical story arcs with evidence from multiple inputs.
    """

    # Common narrative themes in financial research
    THEMES = [
        "growth_trajectory",
        "profitability_trends",
        "risk_factors",
        "competitive_dynamics",
        "management_strategy",
        "market_sentiment",
    ]

    def build_threads(
        self, categorized_data: dict[str, list[dict[str, Any]]]
    ) -> list[NarrativeThread]:
        """Build narrative threads from categorized research data.

        Args:
            categorized_data: Data grouped by source type.

        Returns:
            List of NarrativeThread objects.
        """
        threads = []

        # Growth thread
        growth_thread = self._build_growth_thread(categorized_data)
        if growth_thread.data_points:
            threads.append(growth_thread)

        # Risk thread
        risk_thread = self._build_risk_thread(categorized_data)
        if risk_thread.data_points:
            threads.append(risk_thread)

        # Sentiment thread
        sentiment_thread = self._build_sentiment_thread(categorized_data)
        if sentiment_thread.data_points:
            threads.append(sentiment_thread)

        return threads

    def _build_growth_thread(
        self, data: dict[str, list[dict[str, Any]]]
    ) -> NarrativeThread:
        """Build a narrative about growth trajectory."""
        thread = NarrativeThread(
            theme="growth_trajectory",
            thesis="",
        )

        for source in data.get("financial_data", []):
            source_data = source.get("data", {})
            if isinstance(source_data, dict):
                for key in ("revenue", "revenueGrowth", "revenue_growth"):
                    if key in source_data:
                        thread.data_points.append({
                            "metric": key,
                            "value": source_data[key],
                            "source": source.get("tool", ""),
                        })

        if thread.data_points:
            thread.thesis = "Growth trajectory analysis based on financial data"
            thread.confidence = 0.6 + 0.1 * min(len(thread.data_points), 4)

        return thread

    def _build_risk_thread(
        self, data: dict[str, list[dict[str, Any]]]
    ) -> NarrativeThread:
        """Build a narrative about risk factors."""
        thread = NarrativeThread(
            theme="risk_factors",
            thesis="",
        )

        # SEC filings contain risk factor sections
        for source in data.get("sec_filings", []):
            content = source.get("data", {}).get("content", "")
            if "risk" in content.lower():
                thread.data_points.append({
                    "type": "filing_risk_factors",
                    "source": source.get("tool", ""),
                    "snippet": content[:500],
                })

        # News may contain risk-related information
        for source in data.get("news", []) + data.get("sentiment", []):
            sentiment = source.get("data", {}).get("overall_sentiment", "")
            if sentiment == "negative":
                thread.data_points.append({
                    "type": "negative_sentiment",
                    "source": source.get("tool", ""),
                    "sentiment": sentiment,
                })

        if thread.data_points:
            thread.thesis = "Risk assessment combining filing disclosures and market signals"
            thread.confidence = 0.5 + 0.1 * min(len(thread.data_points), 5)

        return thread

    def _build_sentiment_thread(
        self, data: dict[str, list[dict[str, Any]]]
    ) -> NarrativeThread:
        """Build a narrative about market sentiment."""
        thread = NarrativeThread(
            theme="market_sentiment",
            thesis="",
        )

        for source in data.get("sentiment", []):
            source_data = source.get("data", {})
            if isinstance(source_data, dict):
                thread.data_points.append({
                    "overall": source_data.get("overall_sentiment", "N/A"),
                    "score": source_data.get("average_sentiment_score", 0),
                    "articles": source_data.get("articles_analyzed", 0),
                    "source": source.get("tool", ""),
                })

        if thread.data_points:
            thread.thesis = "Market sentiment analysis from news and media coverage"
            thread.confidence = 0.5 + 0.1 * min(len(thread.data_points), 5)

        return thread

    def format_threads(self, threads: list[NarrativeThread]) -> str:
        """Format narrative threads into readable markdown."""
        if not threads:
            return "No narrative threads could be constructed from available data."

        parts = []
        for thread in threads:
            parts.append(f"### {thread.theme.replace('_', ' ').title()}")
            parts.append(f"*{thread.thesis}* (confidence: {thread.confidence:.0%})")
            parts.append("")
            for dp in thread.data_points:
                source = dp.get("source", "unknown")
                parts.append(f"- {dp} [Source: {source}]")
            parts.append("")

        return "\n".join(parts)
