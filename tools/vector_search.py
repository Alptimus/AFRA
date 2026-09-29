"""
Vector DB Search Tool.

Searches the agent's long-term memory (vector database) for previously
researched information. This tool wraps the memory module's VectorMemory
class as an agent-callable tool.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class VectorDBSearchTool(BaseTool):
    """Search the agent's long-term memory for previously researched information.

    Use this tool BEFORE making external API calls to check if relevant
    information has already been gathered in a previous research session.
    Returns matching document chunks with similarity scores.
    """

    name = "vector_db_search"
    description = (
        "Searches the agent's long-term memory (vector database) for previously "
        "researched information. Use this tool BEFORE making external API calls to "
        "check if relevant data has already been gathered. This reduces redundant "
        "API calls and leverages accumulated research knowledge. Returns relevant "
        "document chunks with similarity scores."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for semantic similarity search",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return (default 5)",
            },
            "filter": {
                "type": "object",
                "description": (
                    "Optional metadata filters (e.g., {'ticker': 'AAPL', "
                    "'source_type': '10-K'} to narrow results)"
                ),
            },
        },
        "required": ["query"],
    }
    fallback_tools = []  # Internal tool — no fallback

    def __init__(
        self, settings: Settings | None = None, vector_memory: Any = None
    ) -> None:
        self._settings = settings or get_settings()
        self._memory = vector_memory  # Injected from memory module

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Search the vector database for relevant documents."""
        query = kwargs["query"]
        top_k = kwargs.get("top_k", 5)
        filters = kwargs.get("filter")

        if not self._memory:
            return ToolResult(
                success=False,
                error="Vector memory not initialized. Memory module must be injected.",
                source=self.name,
            )

        try:
            results = await self._memory.search(
                query=query, top_k=top_k, filters=filters
            )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": [
                        {
                            "content": r.get("content", ""),
                            "similarity_score": r.get("score", 0.0),
                            "metadata": r.get("metadata", {}),
                        }
                        for r in results
                    ],
                    "total_results": len(results),
                },
                source=self.name,
                metadata={"top_k": top_k, "filters_applied": filters is not None},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Vector search error: {type(e).__name__}: {e}",
                source=self.name,
            )
