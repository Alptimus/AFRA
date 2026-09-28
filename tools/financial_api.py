"""
Financial Data API Tool.

Retrieves structured financial data including income statement, balance sheet,
cash flow statement, and key financial ratios. Uses Financial Modeling Prep (FMP)
as the primary API, with yfinance as a fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FinancialDataTool(BaseTool):
    """Retrieve structured financial data for a publicly traded company.

    Use this tool when you need income statement, balance sheet, cash flow,
    or key financial ratios. Returns structured JSON of financial statement data
    for multiple periods.
    """

    name = "financial_data_api"
    description = (
        "Retrieves structured financial data including income statement, balance "
        "sheet, cash flow statement, and key ratios for a publicly traded company. "
        "Use this tool when you need quantitative financial data for analysis, "
        "comparison, or calculation. Returns JSON of financial statement data."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, MSFT)",
            },
            "statement_type": {
                "type": "string",
                "enum": [
                    "income_statement",
                    "balance_sheet",
                    "cash_flow",
                    "ratios",
                    "key_metrics",
                ],
                "description": "Type of financial data to retrieve",
            },
            "period": {
                "type": "string",
                "enum": ["annual", "quarterly"],
                "description": "Reporting period (annual or quarterly)",
            },
            "years": {
                "type": "integer",
                "description": "Number of years/periods of data to retrieve (default 3)",
            },
        },
        "required": ["ticker", "statement_type"],
    }
    fallback_tools = ["sec_filing_search", "web_search", "vector_db_search"]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Retrieve financial data from FMP API or yfinance fallback."""
        ticker = kwargs["ticker"].upper()
        statement_type = kwargs["statement_type"]
        period = kwargs.get("period", "annual")
        years = kwargs.get("years", 3)

        # Try FMP first
        if self._settings.fmp_api_key:
            return await self._fetch_fmp(ticker, statement_type, period, years)

        # Fallback to yfinance
        return await self._fetch_yfinance(ticker, statement_type, period, years)

    async def _fetch_fmp(
        self,
        ticker: str,
        statement_type: str,
        period: str,
        years: int,
    ) -> ToolResult:
        """Fetch from Financial Modeling Prep API."""
        endpoint_map = {
            "income_statement": f"/income-statement/{ticker}",
            "balance_sheet": f"/balance-sheet-statement/{ticker}",
            "cash_flow": f"/cash-flow-statement/{ticker}",
            "ratios": f"/ratios/{ticker}",
            "key_metrics": f"/key-metrics/{ticker}",
        }

        endpoint = endpoint_map.get(statement_type)
        if not endpoint:
            return ToolResult(
                success=False,
                error=f"Unknown statement type: {statement_type}",
                source=self.name,
            )

        url = f"{FMP_BASE_URL}{endpoint}"
        params: dict[str, Any] = {
            "apikey": self._settings.fmp_api_key,
            "limit": years * (4 if period == "quarterly" else 1),
        }
        if statement_type in ("income_statement", "balance_sheet", "cash_flow"):
            params["period"] = period

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, dict) and "Error Message" in data:
                return ToolResult(
                    success=False,
                    error=f"FMP API error: {data['Error Message']}",
                    source=self.name,
                )

            return ToolResult(
                success=True,
                data={
                    "ticker": ticker,
                    "statement_type": statement_type,
                    "period": period,
                    "data": data[:years * (4 if period == "quarterly" else 1)],
                    "periods_returned": len(data) if isinstance(data, list) else 1,
                },
                source=self.name,
                metadata={"provider": "fmp"},
            )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"FMP API HTTP error: {e.response.status_code}",
                source=self.name,
            )
        except httpx.RequestError as e:
            return ToolResult(
                success=False,
                error=f"FMP API request failed: {e}",
                source=self.name,
            )

    async def _fetch_yfinance(
        self,
        ticker: str,
        statement_type: str,
        period: str,
        years: int,
    ) -> ToolResult:
        """Fallback: fetch from yfinance (Yahoo Finance wrapper)."""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            data_map = {
                "income_statement": (
                    stock.financials if period == "annual" else stock.quarterly_financials
                ),
                "balance_sheet": (
                    stock.balance_sheet if period == "annual" else stock.quarterly_balance_sheet
                ),
                "cash_flow": (
                    stock.cashflow if period == "annual" else stock.quarterly_cashflow
                ),
            }

            if statement_type in ("ratios", "key_metrics"):
                # yfinance doesn't have a direct ratios endpoint; compute from data
                info = stock.info
                return ToolResult(
                    success=True,
                    data={
                        "ticker": ticker,
                        "statement_type": statement_type,
                        "data": {
                            "pe_ratio": info.get("trailingPE"),
                            "forward_pe": info.get("forwardPE"),
                            "price_to_book": info.get("priceToBook"),
                            "debt_to_equity": info.get("debtToEquity"),
                            "return_on_equity": info.get("returnOnEquity"),
                            "profit_margins": info.get("profitMargins"),
                            "operating_margins": info.get("operatingMargins"),
                            "revenue_growth": info.get("revenueGrowth"),
                            "earnings_growth": info.get("earningsGrowth"),
                            "current_ratio": info.get("currentRatio"),
                            "quick_ratio": info.get("quickRatio"),
                        },
                    },
                    source=self.name,
                    metadata={"provider": "yfinance"},
                )

            df = data_map.get(statement_type)
            if df is None or df.empty:
                return ToolResult(
                    success=False,
                    error=f"No {statement_type} data available for {ticker} via yfinance",
                    source=self.name,
                )

            # Convert DataFrame to serializable format
            records = []
            for col in df.columns[:years * (4 if period == "quarterly" else 1)]:
                record = {"date": str(col.date()) if hasattr(col, "date") else str(col)}
                for idx in df.index:
                    val = df.at[idx, col]
                    record[str(idx)] = float(val) if val is not None else None
                records.append(record)

            return ToolResult(
                success=True,
                data={
                    "ticker": ticker,
                    "statement_type": statement_type,
                    "period": period,
                    "data": records,
                    "periods_returned": len(records),
                },
                source=self.name,
                metadata={"provider": "yfinance"},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"yfinance error: {type(e).__name__}: {e}",
                source=self.name,
            )
