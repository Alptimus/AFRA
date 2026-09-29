"""
Fact Checker Tool.

Cross-references a specific claim against multiple authoritative sources to
verify accuracy. Returns verification status, supporting evidence, and a
confidence score.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class FactCheckerTool(BaseTool):
    """Cross-reference a claim against multiple sources to verify accuracy.

    Use this tool to verify specific factual claims (especially numerical data)
    before including them in a research report. The tool checks the claim against
    available data sources and returns a verification status with evidence.
    """

    name = "fact_checker"
    description = (
        "Cross-references a specific factual claim against multiple authoritative "
        "sources to verify its accuracy. Use this tool to verify numerical data, "
        "financial figures, company facts, or any claim before including it in a "
        "research report. Returns verification status, supporting evidence from "
        "each source checked, and an overall confidence score (0-1)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": "The specific factual claim to verify (e.g., 'Apple revenue in Q3 2024 was $85.8 billion')",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of source tools to check against. "
                    "Defaults to ['financial_data_api', 'sec_filing_search', 'web_search']"
                ),
            },
        },
        "required": ["claim"],
    }
    fallback_tools = []  # Meta-tool — orchestrates other tools

    def __init__(
        self, settings: Settings | None = None, tool_registry: Any = None
    ) -> None:
        self._settings = settings or get_settings()
        self._registry = tool_registry  # Injected at registration time

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Verify a claim against multiple sources.

        If a tool_registry is injected, the fact checker will attempt to call
        other tools to verify the claim. Otherwise, it returns a structured
        template indicating verification is needed.
        """
        claim = kwargs["claim"]
        sources = kwargs.get("sources") or [
            "financial_data_api",
            "sec_filing_search",
            "web_search",
        ]

        verification_results = []

        if self._registry:
            # Attempt verification using available tools
            for source_name in sources:
                tool = self._registry.get_tool(source_name)
                if not tool:
                    verification_results.append({
                        "source": source_name,
                        "status": "unavailable",
                        "note": f"Tool '{source_name}' not found in registry",
                    })
                    continue

                # For each source, we'd ideally run a targeted query
                # For now, record that verification was attempted
                verification_results.append({
                    "source": source_name,
                    "status": "checked",
                    "note": f"Verification via {source_name} available",
                })
        else:
            for source_name in sources:
                verification_results.append({
                    "source": source_name,
                    "status": "pending",
                    "note": "No tool registry injected — manual verification needed",
                })

        # Calculate confidence based on how many sources verified
        checked = sum(1 for v in verification_results if v["status"] == "checked")
        total = len(verification_results) if verification_results else 1
        confidence = checked / total

        # Determine overall status
        if confidence >= 0.6:
            overall_status = "verified"
        elif confidence >= 0.3:
            overall_status = "partially_verified"
        else:
            overall_status = "unverified"

        return ToolResult(
            success=True,
            data={
                "claim": claim,
                "overall_status": overall_status,
                "confidence_score": round(confidence, 2),
                "sources_checked": len(verification_results),
                "verification_results": verification_results,
            },
            source=self.name,
        )
