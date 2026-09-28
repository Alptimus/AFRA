"""
SEC EDGAR Filing Search Tool.

Retrieves SEC filings (10-K, 10-Q, 8-K, DEF 14A) from the EDGAR full-text
search system (efts.sec.gov). Free API, no key required — only a User-Agent
header identifying the application.

Rate limit: 10 requests per second (SEC policy).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# SEC EDGAR EFTS base URL
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_EFTS_API = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL_SEARCH = "https://efts.sec.gov/LATEST/search-index"

# Simpler, more reliable endpoints
EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


class SecFilingSearchTool(BaseTool):
    """Search and retrieve SEC EDGAR filings for a publicly traded US company.

    Use this tool when you need official regulatory disclosures including
    annual reports (10-K), quarterly reports (10-Q), material event reports
    (8-K), or proxy statements (DEF 14A). Returns filing metadata and text.
    """

    name = "sec_filing_search"
    description = (
        "Search and retrieve SEC EDGAR filings for a publicly traded US company. "
        "Use this tool when you need official regulatory disclosures including "
        "annual reports (10-K), quarterly reports (10-Q), material event reports "
        "(8-K), or proxy statements (DEF 14A). Returns filing text and metadata."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
            },
            "filing_type": {
                "type": "string",
                "enum": ["10-K", "10-Q", "8-K", "DEF 14A"],
                "description": "Type of SEC filing to retrieve",
            },
            "year": {
                "type": "integer",
                "description": "Filing year (defaults to most recent if omitted)",
            },
        },
        "required": ["ticker", "filing_type"],
    }
    fallback_tools = ["web_search", "vector_db_search"]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self._settings.sec_edgar_user_agent},
            timeout=30.0,
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Retrieve SEC filing data from EDGAR.

        Uses the SEC EDGAR full-text search API to find and retrieve filings.
        """
        ticker = kwargs["ticker"].upper()
        filing_type = kwargs["filing_type"]
        year = kwargs.get("year")

        try:
            # Step 1: Look up the company CIK from ticker
            cik = await self._get_cik(ticker)
            if not cik:
                return ToolResult(
                    success=False,
                    error=f"Could not find CIK for ticker '{ticker}'. "
                    "Ensure this is a valid US-listed company ticker.",
                    source=self.name,
                )

            # Step 2: Get recent filings for this company
            filings = await self._get_filings(cik, filing_type, year)
            if not filings:
                return ToolResult(
                    success=False,
                    error=f"No {filing_type} filings found for {ticker}"
                    + (f" in {year}" if year else ""),
                    source=self.name,
                )

            # Step 3: Get the filing content (first match)
            filing = filings[0]
            content = await self._get_filing_content(filing.get("url", ""))

            return ToolResult(
                success=True,
                data={
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "filing_date": filing.get("date", ""),
                    "accession_number": filing.get("accession_number", ""),
                    "content": content[:15000] if content else "",  # Truncate for context
                    "url": filing.get("url", ""),
                    "total_filings_found": len(filings),
                },
                source=self.name,
                metadata={
                    "cik": cik,
                    "filing_count": len(filings),
                    "truncated": len(content) > 15000 if content else False,
                },
            )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"SEC EDGAR API HTTP error: {e.response.status_code}",
                source=self.name,
            )
        except httpx.RequestError as e:
            return ToolResult(
                success=False,
                error=f"SEC EDGAR API request failed: {e}",
                source=self.name,
            )

    async def _get_cik(self, ticker: str) -> str | None:
        """Look up a company's CIK number from its ticker symbol."""
        try:
            resp = await self._client.get(EDGAR_COMPANY_TICKERS_URL)
            resp.raise_for_status()
            data = resp.json()

            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"])
            return None
        except Exception as e:
            logger.warning("CIK lookup failed for %s: %s", ticker, e)
            return None

    async def _get_filings(
        self, cik: str, filing_type: str, year: int | None = None
    ) -> list[dict[str, Any]]:
        """Get a list of filings for a company from SEC EDGAR."""
        try:
            url = EDGAR_SUBMISSIONS_URL.format(cik=int(cik))
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            filings = []
            for i, form in enumerate(forms):
                if form != filing_type:
                    continue
                filing_date = dates[i] if i < len(dates) else ""
                if year and not filing_date.startswith(str(year)):
                    continue

                accession = accessions[i] if i < len(accessions) else ""
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""
                accession_path = accession.replace("-", "")

                filings.append({
                    "form": form,
                    "date": filing_date,
                    "accession_number": accession,
                    "url": (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{int(cik)}/{accession_path}/{primary_doc}"
                    ),
                })

            return filings[:5]  # Return up to 5 most recent

        except Exception as e:
            logger.warning("Filing search failed for CIK %s: %s", cik, e)
            return []

    async def _get_filing_content(self, url: str) -> str:
        """Download and extract text content from a filing URL."""
        if not url:
            return ""
        try:
            resp = await self._client.get(url, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text

            # Basic HTML stripping for readable text
            if "<html" in text.lower() or "<body" in text.lower():
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(text, "html.parser")
                # Remove scripts and styles
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)

            return text
        except Exception as e:
            logger.warning("Failed to fetch filing content from %s: %s", url, e)
            return ""

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
