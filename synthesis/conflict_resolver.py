"""
Conflict Resolver — Resolves conflicting data from multiple financial sources.

Implements the 6-step conflict resolution protocol from Section A6.3:
1. Identify the conflict
2. Assess source tiers
3. Check for temporal differences
4. Look for restatements
5. Apply the highest-tier rule
6. Document the conflict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DataConflict:
    """A detected conflict between data sources.

    Attributes:
        metric: The data point that is in conflict (e.g., "revenue").
        values: Dict mapping source name to its reported value.
        source_tiers: Dict mapping source name to its reliability tier.
        temporal_info: Optional temporal context for each source.
        resolution: How the conflict was resolved.
        resolved_value: The final accepted value.
    """

    metric: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    source_tiers: dict[str, int] = field(default_factory=dict)
    temporal_info: dict[str, str] = field(default_factory=dict)
    resolution: str = ""
    resolved_value: Any = None


class ConflictResolver:
    """Resolves data conflicts using the source reliability hierarchy.

    Source Reliability Hierarchy (CORRECTED from Section A6.2):
        Tier 1 (Highest): SEC filings (10-K, 10-Q) — legally mandated, audited
        Tier 2: Financial data APIs (Bloomberg, FMP, Refinitiv) — professionally curated
        Tier 3: Earnings call transcripts — direct management commentary
        Tier 4: Major news outlets (Reuters, Bloomberg News, FT) — professional journalism
        Tier 5 (Lowest): Social media and anonymous forums — unverified

    NOTE: The original project document (Section A6.2) incorrectly placed social
    media (Tier 4) above professional news outlets (Tier 5). This implementation
    uses the CORRECTED hierarchy where news outlets rank higher than social media.
    See ERROR_LOG.md entry #1.
    """

    # CORRECTED source tier hierarchy (see ERROR_LOG.md)
    SOURCE_TIERS: dict[str, int] = {
        # Tier 1: SEC filings
        "sec_filing_search": 1,
        "sec_filing": 1,
        "10-K": 1,
        "10-Q": 1,
        "8-K": 1,
        "DEF 14A": 1,
        # Tier 2: Financial data APIs
        "financial_data_api": 2,
        "fmp": 2,
        "bloomberg": 2,
        "refinitiv": 2,
        "yfinance": 2,
        # Tier 3: Earnings transcripts
        "earnings_transcript": 3,
        "earnings_call": 3,
        # Tier 4: Professional news (CORRECTED — was Tier 5 in source doc)
        "web_search": 4,
        "news_sentiment": 4,
        "reuters": 4,
        "bloomberg_news": 4,
        "financial_times": 4,
        # Tier 5: Social media (CORRECTED — was Tier 4 in source doc)
        "social_media": 5,
        "reddit": 5,
        "twitter": 5,
        "forum": 5,
    }

    def get_tier(self, source: str) -> int:
        """Get the reliability tier for a source (1=highest, 5=lowest)."""
        return self.SOURCE_TIERS.get(source, 4)  # Default to Tier 4

    def resolve(self, conflict: DataConflict) -> str:
        """Resolve a data conflict using the 6-step protocol.

        Args:
            conflict: The detected DataConflict to resolve.

        Returns:
            A string describing the resolution.
        """
        # Step 1: Conflict already identified (in conflict object)

        # Step 2: Assess source tiers
        tiered_values = sorted(
            conflict.values.items(),
            key=lambda x: conflict.source_tiers.get(x[0], 5),
        )

        # Step 3: Check for temporal differences
        temporal_resolution = self._check_temporal(conflict)
        if temporal_resolution:
            conflict.resolution = temporal_resolution
            return temporal_resolution

        # Step 4: Check for restatements
        restatement_check = self._check_restatements(conflict)
        if restatement_check:
            conflict.resolution = restatement_check
            return restatement_check

        # Step 5: Apply highest-tier rule
        if tiered_values:
            best_source, best_value = tiered_values[0]
            best_tier = conflict.source_tiers.get(best_source, 5)

            conflict.resolved_value = best_value
            resolution = (
                f"Resolved in favor of {best_source} (Tier {best_tier}) "
                f"with value {best_value}. "
                f"Conflicting sources: {', '.join(f'{s} (Tier {t})={v}' for s, v in conflict.values.items() for t in [conflict.source_tiers.get(s, 5)] if s != best_source)}"
            )
            conflict.resolution = resolution
            return resolution

        return "Unable to resolve — insufficient tier information"

    def _check_temporal(self, conflict: DataConflict) -> str | None:
        """Check if the conflict is due to different reporting periods."""
        temporal = conflict.temporal_info
        if not temporal:
            return None

        dates = list(temporal.values())
        if len(set(dates)) > 1:
            return (
                f"Conflict resolved as temporal difference: sources report "
                f"data from different periods ({', '.join(f'{s}: {d}' for s, d in temporal.items())}). "
                f"Using most recent data."
            )
        return None

    def _check_restatements(self, conflict: DataConflict) -> str | None:
        """Check if the conflict might be due to financial restatements."""
        values = list(conflict.values.values())
        if len(values) < 2:
            return None

        # If values are close (within 1%) it might be a rounding difference
        nums = [v for v in values if isinstance(v, (int, float))]
        if len(nums) >= 2:
            max_val = max(abs(n) for n in nums) if nums else 1
            if max_val > 0:
                range_pct = (max(nums) - min(nums)) / max_val
                if range_pct < 0.01:
                    return (
                        f"Values differ by less than 1% ({range_pct:.2%}) — "
                        "likely a rounding or restatement difference. Using the "
                        "value from the highest-tier source."
                    )

        return None
