"""
Peer Comparison Tool.

Identifies peer companies and retrieves comparative financial metrics.
Uses FMP API for peer identification and financial data comparison.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

FMP_PEERS_URL = "https://financialmodelingprep.com/api/v4/stock_peers"
FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile/{ticker}"
FMP_RATIOS_TTM_URL = "https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}"


class PeerComparisonTool(BaseTool):
    """Identify peer companies and compare financial metrics.

    Use this tool when you need to understand a company's competitive position
    relative to its industry peers. Returns a comparison matrix ranking companies
    across specified financial metrics.
    """

    name = "peer_comparison"
    description = (
        "Identifies peer companies in the same industry and retrieves comparative "
        "financial metrics. Use this tool for competitive analysis, relative "
        "valuation, or industry benchmarking. Returns a peer comparison matrix "
        "with rankings across financial metrics."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol of the target company",
            },
            "num_peers": {
                "type": "integer",
                "description": "Number of peer companies to compare (default 5)",
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of metrics to compare (e.g., 'market_cap', 'pe_ratio', "
                    "'revenue_growth', 'profit_margin', 'roe'). Defaults to standard set."
                ),
            },
        },
        "required": ["ticker"],
    }
    fallback_tools = ["web_search", "calculation_engine"]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Identify peers and compare metrics."""
        ticker = kwargs["ticker"].upper()
        num_peers = kwargs.get("num_peers", 5)
        metrics = kwargs.get("metrics") or [
            "market_cap",
            "pe_ratio",
            "price_to_book",
            "roe",
            "profit_margin",
            "revenue_growth",
            "debt_to_equity",
        ]

        if self._settings.fmp_api_key:
            return await self._compare_fmp(ticker, num_peers, metrics)
        return await self._compare_yfinance(ticker, num_peers, metrics)

    async def _compare_fmp(
        self, ticker: str, num_peers: int, metrics: list[str]
    ) -> ToolResult:
        """Peer comparison using FMP API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get peer list
                resp = await client.get(
                    FMP_PEERS_URL,
                    params={"symbol": ticker, "apikey": self._settings.fmp_api_key},
                )
                resp.raise_for_status()
                peers_data = resp.json()

            peer_list = []
            if peers_data and isinstance(peers_data, list):
                peer_list = peers_data[0].get("peersList", [])[:num_peers]
            elif peers_data and isinstance(peers_data, dict):
                peer_list = peers_data.get("peersList", [])[:num_peers]

            if not peer_list:
                return ToolResult(
                    success=False,
                    error=f"No peers found for {ticker}",
                    source=self.name,
                )

            # Get comparison data for target + peers
            all_tickers = [ticker] + peer_list
            comparison = []

            async with httpx.AsyncClient(timeout=30.0) as client:
                for t in all_tickers:
                    profile_resp = await client.get(
                        FMP_PROFILE_URL.format(ticker=t),
                        params={"apikey": self._settings.fmp_api_key},
                    )
                    if profile_resp.status_code != 200:
                        continue
                    profile = profile_resp.json()
                    if not profile:
                        continue
                    p = profile[0] if isinstance(profile, list) else profile

                    company_data = {
                        "ticker": t,
                        "company_name": p.get("companyName", ""),
                        "market_cap": p.get("mktCap"),
                        "price": p.get("price"),
                        "sector": p.get("sector", ""),
                        "industry": p.get("industry", ""),
                        "is_target": t == ticker,
                    }
                    comparison.append(company_data)

            return ToolResult(
                success=True,
                data={
                    "target_ticker": ticker,
                    "peers": peer_list,
                    "comparison": comparison,
                    "metrics_requested": metrics,
                },
                source=self.name,
                metadata={"provider": "fmp", "peers_found": len(peer_list)},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"FMP peer comparison error: {e}",
                source=self.name,
            )

    async def _compare_yfinance(
        self, ticker: str, num_peers: int, metrics: list[str]
    ) -> ToolResult:
        """Fallback peer comparison using yfinance."""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get("sector", "")
            industry = info.get("industry", "")

            target_data = {
                "ticker": ticker,
                "company_name": info.get("longName", ""),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "price_to_book": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "sector": sector,
                "industry": industry,
                "is_target": True,
            }

            return ToolResult(
                success=True,
                data={
                    "target_ticker": ticker,
                    "peers": [],
                    "comparison": [target_data],
                    "metrics_requested": metrics,
                    "note": "yfinance fallback — peer identification requires FMP API key",
                },
                source=self.name,
                metadata={"provider": "yfinance"},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"yfinance peer comparison error: {e}",
                source=self.name,
            )
