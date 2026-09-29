"""
Vector DB Store Tool.

Stores new research findings in the agent's long-term memory (vector database)
for future retrieval. This tool wraps the memory module's VectorMemory class.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class VectorDBStoreTool(BaseTool):
    """Store new research findings in the agent's long-term memory.

    Use this tool to persist important research findings, key data points,
    and analysis conclusions for future retrieval. Include metadata such as
    ticker, source type, and date for effective filtering.
    """

    name = "vector_db_store"
    description = (
        "Stores new research findings in the agent's long-term memory (vector "
        "database) for future retrieval. Use this tool to persist important "
        "findings, data points, and analysis conclusions after completing research. "
        "Include metadata (ticker, date, source_type) for effective future filtering."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text content to store in memory",
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Metadata for the content. Should include: ticker (str), "
                    "date (str), source_type (str: '10-K', 'earnings_call', "
                    "'news', 'analysis'), confidence (float 0-1)"
                ),
            },
        },
        "required": ["content", "metadata"],
    }
    fallback_tools = []  # Internal tool — no fallback

    def __init__(
        self, settings: Settings | None = None, vector_memory: Any = None
    ) -> None:
        self._settings = settings or get_settings()
        self._memory = vector_memory  # Injected from memory module

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Store content in the vector database."""
        content = kwargs["content"]
        metadata = kwargs.get("metadata", {})

        if not self._memory:
            return ToolResult(
                success=False,
                error="Vector memory not initialized. Memory module must be injected.",
                source=self.name,
            )

        try:
            doc_id = await self._memory.store(content=content, metadata=metadata)

            return ToolResult(
                success=True,
                data={
                    "document_id": doc_id,
                    "content_length": len(content),
                    "metadata": metadata,
                    "status": "stored",
                },
                source=self.name,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Vector store error: {type(e).__name__}: {e}",
                source=self.name,
            )
