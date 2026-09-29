"""
Company Profile Tool.

Retrieves basic company information including sector, industry, market cap,
executives, and description. Uses FMP API or yfinance as fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile/{ticker}"


class CompanyProfileTool(BaseTool):
    """Retrieve basic company profile information.

    Use this tool to get an overview of a company including its sector, industry,
    market capitalization, key executives, business description, and contact
    information. Good starting point for any company research.
    """

    name = "company_profile"
    description = (
        "Retrieves basic company information including sector, industry, market "
        "capitalization, CEO, number of employees, and business description. "
        "Use this tool as a starting point for company research or when you need "
        "to classify a company's business segment and competitive context."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, MSFT)",
            },
        },
        "required": ["ticker"],
    }
    fallback_tools = ["web_search", "vector_db_search"]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Retrieve company profile from FMP or yfinance."""
        ticker = kwargs["ticker"].upper()

        if self._settings.fmp_api_key:
            return await self._fetch_fmp(ticker)
        return await self._fetch_yfinance(ticker)

    async def _fetch_fmp(self, ticker: str) -> ToolResult:
        """Fetch profile from FMP API."""
        url = FMP_PROFILE_URL.format(ticker=ticker)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    url, params={"apikey": self._settings.fmp_api_key}
                )
                resp.raise_for_status()
                data = resp.json()

            if not data:
                return ToolResult(
                    success=False,
                    error=f"No profile found for ticker '{ticker}'",
                    source=self.name,
                )

            profile = data[0] if isinstance(data, list) else data
            return ToolResult(
                success=True,
                data={
                    "ticker": ticker,
                    "company_name": profile.get("companyName", ""),
                    "sector": profile.get("sector", ""),
                    "industry": profile.get("industry", ""),
                    "market_cap": profile.get("mktCap"),
                    "price": profile.get("price"),
                    "beta": profile.get("beta"),
                    "vol_avg": profile.get("volAvg"),
                    "ceo": profile.get("ceo", ""),
                    "employees": profile.get("fullTimeEmployees"),
                    "headquarters": f"{profile.get('city', '')}, {profile.get('state', '')}, {profile.get('country', '')}",
                    "website": profile.get("website", ""),
                    "description": profile.get("description", ""),
                    "exchange": profile.get("exchangeShortName", ""),
                    "ipo_date": profile.get("ipoDate", ""),
                    "currency": profile.get("currency", "USD"),
                },
                source=self.name,
                metadata={"provider": "fmp"},
            )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"FMP profile API error: {e.response.status_code}",
                source=self.name,
            )
        except httpx.RequestError as e:
            return ToolResult(
                success=False,
                error=f"FMP profile request failed: {e}",
                source=self.name,
            )

    async def _fetch_yfinance(self, ticker: str) -> ToolResult:
        """Fallback: fetch profile from yfinance."""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or "symbol" not in info:
                return ToolResult(
                    success=False,
                    error=f"No profile data found for '{ticker}' via yfinance",
                    source=self.name,
                )

            return ToolResult(
                success=True,
                data={
                    "ticker": ticker,
                    "company_name": info.get("longName", info.get("shortName", "")),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "market_cap": info.get("marketCap"),
                    "price": info.get("currentPrice", info.get("regularMarketPrice")),
                    "beta": info.get("beta"),
                    "vol_avg": info.get("averageVolume"),
                    "ceo": "",  # yfinance doesn't provide CEO directly
                    "employees": info.get("fullTimeEmployees"),
                    "headquarters": f"{info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}",
                    "website": info.get("website", ""),
                    "description": info.get("longBusinessSummary", ""),
                    "exchange": info.get("exchange", ""),
                    "currency": info.get("currency", "USD"),
                },
                source=self.name,
                metadata={"provider": "yfinance"},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"yfinance profile error: {type(e).__name__}: {e}",
                source=self.name,
            )
