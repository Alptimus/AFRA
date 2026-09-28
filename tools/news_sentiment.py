"""
News Sentiment Analysis Tool.

Analyzes sentiment of recent news articles about a company or financial topic.
Uses NewsAPI.org for article retrieval and TextBlob for sentiment scoring.
Falls back to web search + TextBlob when NewsAPI is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsSentimentTool(BaseTool):
    """Analyze sentiment of recent news articles about a company or topic.

    Use this tool when you need to gauge market sentiment, media coverage tone,
    or public perception of a company or financial event. Returns sentiment
    scores, article summaries, and overall sentiment assessment.
    """

    name = "news_sentiment"
    description = (
        "Analyzes sentiment of recent news articles about a company or financial "
        "topic using NLP. Use this tool when you need to understand market "
        "sentiment, media coverage tone, or public perception. Returns individual "
        "article sentiment scores, summaries, and an overall sentiment assessment."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Company name, ticker, or topic to analyze sentiment for",
            },
            "num_articles": {
                "type": "integer",
                "description": "Number of articles to analyze (default 10)",
            },
            "lookback_days": {
                "type": "integer",
                "description": "Number of days to look back for articles (default 30)",
            },
        },
        "required": ["query"],
    }
    fallback_tools = ["web_search"]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Retrieve news articles and compute sentiment scores."""
        query = kwargs["query"]
        num_articles = kwargs.get("num_articles", 10)
        lookback_days = kwargs.get("lookback_days", 30)

        # Try NewsAPI first, then fall back to web-search-based approach
        if self._settings.newsapi_key:
            articles = await self._fetch_newsapi(query, num_articles, lookback_days)
        else:
            # Use web search as article source
            articles = await self._fetch_via_websearch(query, num_articles)

        if not articles:
            return ToolResult(
                success=False,
                error=f"No news articles found for query: '{query}'",
                source=self.name,
            )

        # Analyze sentiment for each article
        analyzed = self._analyze_sentiment(articles)

        # Compute overall sentiment
        sentiments = [a["sentiment_score"] for a in analyzed]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        overall = "neutral"
        if avg_sentiment > 0.1:
            overall = "positive"
        elif avg_sentiment < -0.1:
            overall = "negative"

        return ToolResult(
            success=True,
            data={
                "query": query,
                "overall_sentiment": overall,
                "average_sentiment_score": round(avg_sentiment, 4),
                "articles_analyzed": len(analyzed),
                "sentiment_distribution": {
                    "positive": sum(1 for s in sentiments if s > 0.1),
                    "neutral": sum(1 for s in sentiments if -0.1 <= s <= 0.1),
                    "negative": sum(1 for s in sentiments if s < -0.1),
                },
                "articles": analyzed[:num_articles],
            },
            source=self.name,
        )

    async def _fetch_newsapi(
        self, query: str, num_articles: int, lookback_days: int
    ) -> list[dict[str, Any]]:
        """Fetch articles from NewsAPI.org."""
        from datetime import datetime, timedelta

        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    NEWSAPI_URL,
                    params={
                        "q": query,
                        "from": from_date,
                        "sortBy": "relevancy",
                        "pageSize": min(num_articles, 100),
                        "apiKey": self._settings.newsapi_key,
                        "language": "en",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "published_at": item.get("publishedAt", ""),
                })
            return articles

        except Exception as e:
            logger.warning("NewsAPI request failed: %s", e)
            return []

    async def _fetch_via_websearch(
        self, query: str, num_articles: int
    ) -> list[dict[str, Any]]:
        """Fallback: get articles via Tavily web search."""
        if not self._settings.tavily_api_key:
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._settings.tavily_api_key,
                        "query": f"{query} news",
                        "max_results": num_articles,
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            articles = []
            for item in data.get("results", []):
                articles.append({
                    "title": item.get("title", ""),
                    "description": item.get("content", "")[:300],
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "source": "web_search",
                    "published_at": "",
                })
            return articles

        except Exception as e:
            logger.warning("Web search fallback for news failed: %s", e)
            return []

    def _analyze_sentiment(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute sentiment scores using TextBlob."""
        try:
            from textblob import TextBlob
        except ImportError:
            logger.warning("TextBlob not available, returning neutral sentiments")
            for article in articles:
                article["sentiment_score"] = 0.0
                article["sentiment_label"] = "neutral"
            return articles

        analyzed = []
        for article in articles:
            text = f"{article.get('title', '')}. {article.get('description', '')}"
            if not text.strip(". "):
                continue

            blob = TextBlob(text)
            score = blob.sentiment.polarity  # -1 to 1

            label = "neutral"
            if score > 0.1:
                label = "positive"
            elif score < -0.1:
                label = "negative"

            analyzed.append({
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "published_at": article.get("published_at", ""),
                "sentiment_score": round(score, 4),
                "sentiment_label": label,
                "summary": article.get("description", "")[:200],
            })

        return analyzed
