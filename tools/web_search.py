"""
Web Search Tool.

Performs web searches for current news, analysis, and commentary using
the Tavily API (primary) or a fallback scraping approach.

Tavily is an AI-optimized search API that returns clean, structured results
ideal for LLM consumption.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchTool(BaseTool):
    """Perform web search for current news, analysis, and commentary.

    Use this tool when you need real-time information about a company, market
    event, or financial topic that may not be available in SEC filings or
    financial databases. Returns a list of relevant URLs with titles and snippets.
    """

    name = "web_search"
    description = (
        "Performs web search for current news, analysis, and commentary about "
        "a company or financial topic. Use this tool when you need real-time "
        "information, recent developments, analyst opinions, or market commentary "
        "that may not be in regulatory filings or structured databases. Returns "
        "a list of URL, title, and content snippet tuples."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 10)",
            },
            "date_range": {
                "type": "string",
                "description": "Optional date range filter (e.g., 'past_week', 'past_month', 'past_year')",
            },
        },
        "required": ["query"],
    }
    fallback_tools = ["news_sentiment", "vector_db_search"]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Execute a web search via Tavily API."""
        query = kwargs["query"]
        num_results = kwargs.get("num_results", 10)
        date_range = kwargs.get("date_range")

        api_key = self._settings.tavily_api_key
        if not api_key:
            return ToolResult(
                success=False,
                error="Tavily API key not configured. Set TAVILY_API_KEY in .env",
                source=self.name,
            )

        try:
            payload: dict[str, Any] = {
                "api_key": api_key,
                "query": query,
                "max_results": min(num_results, 20),
                "include_answer": True,
                "include_raw_content": False,
                "search_depth": "advanced",
            }

            if date_range:
                payload["days"] = {
                    "past_week": 7,
                    "past_month": 30,
                    "past_year": 365,
                }.get(date_range, 30)

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(TAVILY_SEARCH_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:500],
                    "score": item.get("score", 0.0),
                })

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "answer": data.get("answer", ""),
                    "results": results,
                    "total_results": len(results),
                },
                source=self.name,
                metadata={"search_depth": "advanced"},
            )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"Tavily API error: {e.response.status_code} — {e.response.text[:200]}",
                source=self.name,
            )
        except httpx.RequestError as e:
            return ToolResult(
                success=False,
                error=f"Web search request failed: {e}",
                source=self.name,
            )
