"""
Fallback Chains — Defines fallback tool sequences for graceful degradation.

When a primary tool fails, the agent tries each fallback in order until
one succeeds or all options are exhausted.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Fallback Chain Definitions ────────────────────────────────────────────────

FALLBACK_CHAINS: dict[str, list[str]] = {
    # Primary tool → ordered list of fallbacks
    "sec_filing_search": ["web_search", "vector_db_search"],
    "financial_data_api": ["sec_filing_search", "web_search", "vector_db_search"],
    "web_search": ["news_sentiment", "vector_db_search"],
    "earnings_transcript": ["web_search", "vector_db_search"],
    "news_sentiment": ["web_search"],
    "company_profile": ["web_search", "vector_db_search"],
    "peer_comparison": ["web_search", "calculation_engine"],
    # Internal tools — no external fallbacks needed
    "vector_db_search": [],
    "vector_db_store": [],
    "report_generator": [],
    "fact_checker": [],
    "calculation_engine": [],
}


def get_fallback_chain(tool_name: str) -> list[str]:
    """Get the fallback chain for a given tool.

    Args:
        tool_name: Name of the primary tool.

    Returns:
        Ordered list of fallback tool names (empty if no fallbacks defined).
    """
    return FALLBACK_CHAINS.get(tool_name, [])


def get_all_chains() -> dict[str, list[str]]:
    """Get all defined fallback chains."""
    return FALLBACK_CHAINS.copy()
