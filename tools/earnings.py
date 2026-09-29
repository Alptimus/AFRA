"""
Earnings Transcript Tool.

Retrieves earnings call transcripts for a specific company, quarter, and year.
Uses Financial Modeling Prep (FMP) API as primary source, with web search fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

FMP_TRANSCRIPT_URL = "https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"


class EarningsTranscriptTool(BaseTool):
    """Retrieve earnings call transcript for a specific company and quarter.

    Use this tool when you need direct management commentary, forward-looking
    guidance, analyst Q&A, or qualitative insights about a company's strategy
    and outlook. Returns the full transcript text with speaker labels.
    """

    name = "earnings_transcript"
    description = (
        "Retrieves earnings call transcript for a specific company, quarter, and "
        "year. Use this tool when you need management commentary, forward guidance, "
        "analyst Q&A, or qualitative insights about strategy and operations. "
        "Returns the full transcript text with speaker labels."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, MSFT)",
            },
            "quarter": {
                "type": "string",
                "enum": ["Q1", "Q2", "Q3", "Q4"],
                "description": "Fiscal quarter (Q1 through Q4)",
            },
            "year": {
                "type": "integer",
                "description": "Fiscal year for the earnings call",
            },
        },
        "required": ["ticker", "quarter", "year"],
    }
    fallback_tools = ["web_search", "vector_db_search"]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Retrieve earnings transcript from FMP or fallback."""
        ticker = kwargs["ticker"].upper()
        quarter = kwargs["quarter"]
        year = kwargs["year"]

        if self._settings.fmp_api_key:
            return await self._fetch_fmp(ticker, quarter, year)

        return ToolResult(
            success=False,
            error=(
                "No FMP API key configured for earnings transcripts. "
                "Set FMP_API_KEY in .env or use web_search as fallback."
            ),
            source=self.name,
        )

    async def _fetch_fmp(
        self, ticker: str, quarter: str, year: int
    ) -> ToolResult:
        """Fetch transcript from Financial Modeling Prep."""
        # Map quarter to FMP's expected format
        quarter_num = int(quarter[1])  # Q1 → 1, Q2 → 2, etc.

        url = FMP_TRANSCRIPT_URL.format(ticker=ticker)
        params = {
            "quarter": quarter_num,
            "year": year,
            "apikey": self._settings.fmp_api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if not data:
                return ToolResult(
                    success=False,
                    error=f"No earnings transcript found for {ticker} {quarter} {year}",
                    source=self.name,
                )

            transcript = data[0] if isinstance(data, list) else data
            content = transcript.get("content", "")

            return ToolResult(
                success=True,
                data={
                    "ticker": ticker,
                    "quarter": quarter,
                    "year": year,
                    "date": transcript.get("date", ""),
                    "content": content[:20000],  # Truncate for context window
                    "content_length": len(content),
                },
                source=self.name,
                metadata={
                    "provider": "fmp",
                    "truncated": len(content) > 20000,
                },
            )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"FMP transcript API error: {e.response.status_code}",
                source=self.name,
            )
        except httpx.RequestError as e:
            return ToolResult(
                success=False,
                error=f"FMP transcript request failed: {e}",
                source=self.name,
            )
