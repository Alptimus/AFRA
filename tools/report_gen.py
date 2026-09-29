"""
Report Generator Tool.

Formats researched data into a structured investment research report following
a specified template. Uses the LLM to generate professional prose from
structured data inputs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from config.settings import Settings, get_settings
from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ── Report Templates ──────────────────────────────────────────────────────────

REPORT_TEMPLATES = {
    "full_research": {
        "sections": [
            "Executive Summary",
            "Company Overview",
            "Financial Analysis",
            "Risk Assessment",
            "Competitive Position",
            "Industry Outlook",
            "Research Methodology Notes",
        ],
        "description": "Comprehensive investment research report",
    },
    "risk_assessment": {
        "sections": [
            "Executive Summary",
            "Risk Factor Overview",
            "Operational Risks",
            "Financial Risks",
            "Market and Competitive Risks",
            "Regulatory and Legal Risks",
            "Risk Mitigation Factors",
            "Methodology Notes",
        ],
        "description": "Focused risk assessment report",
    },
    "competitive_analysis": {
        "sections": [
            "Executive Summary",
            "Industry Overview",
            "Competitive Landscape",
            "Company Positioning",
            "Peer Comparison Matrix",
            "Competitive Advantages and Vulnerabilities",
            "Outlook",
            "Methodology Notes",
        ],
        "description": "Competitive positioning and peer analysis report",
    },
    "earnings_review": {
        "sections": [
            "Executive Summary",
            "Financial Highlights",
            "Segment Analysis",
            "Management Commentary Key Takeaways",
            "Guidance and Outlook",
            "Analyst Q&A Highlights",
            "Methodology Notes",
        ],
        "description": "Earnings call review and analysis report",
    },
    "company_profile": {
        "sections": [
            "Company Overview",
            "Business Description",
            "Key Financial Metrics",
            "Recent Developments",
            "Investment Highlights",
            "Methodology Notes",
        ],
        "description": "Company profile and overview report",
    },
}


class ReportGeneratorTool(BaseTool):
    """Format researched data into a structured investment research report.

    Use this tool after gathering and synthesizing data from multiple sources
    to produce the final formatted report. The tool organizes findings into
    a professional research report following the specified template.
    """

    name = "report_generator"
    description = (
        "Formats researched data into a structured investment research report "
        "following a specified template. Use this tool as the final step after "
        "data gathering and synthesis to produce the formatted output. Supported "
        "templates: 'full_research', 'risk_assessment', 'competitive_analysis', "
        "'earnings_review', 'company_profile'. Returns a formatted markdown report."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "enum": list(REPORT_TEMPLATES.keys()),
                "description": "Report template to use",
            },
            "sections": {
                "type": "object",
                "description": (
                    "Dict mapping section names to their content. "
                    "Each key should match a section in the template."
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of data sources used in the report",
            },
        },
        "required": ["template", "sections"],
    }
    fallback_tools = []

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Generate a formatted research report from structured data."""
        template_name = kwargs["template"]
        sections_data = kwargs["sections"]
        sources = kwargs.get("sources", [])

        template = REPORT_TEMPLATES.get(template_name)
        if not template:
            return ToolResult(
                success=False,
                error=f"Unknown template: '{template_name}'. "
                f"Available: {list(REPORT_TEMPLATES.keys())}",
                source=self.name,
            )

        # Build the report
        report_lines = []
        report_lines.append(f"# {template['description'].title()}")
        report_lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # Assemble sections
        sections_completed = 0
        sections_missing = []

        for section_name in template["sections"]:
            report_lines.append(f"## {section_name}")
            report_lines.append("")

            content = sections_data.get(section_name, "")
            if not content:
                # Check alternative keys (lowercase, underscored)
                alt_key = section_name.lower().replace(" ", "_")
                content = sections_data.get(alt_key, "")

            if content:
                report_lines.append(str(content))
                sections_completed += 1
            else:
                report_lines.append(
                    f"*[Section not completed — insufficient data available]*"
                )
                sections_missing.append(section_name)

            report_lines.append("")

        # Sources section
        if sources:
            report_lines.append("## Sources")
            report_lines.append("")
            for i, source in enumerate(sources, 1):
                report_lines.append(f"{i}. {source}")
            report_lines.append("")

        # Disclaimer
        report_lines.append("---")
        report_lines.append(
            "*This report was generated by AFRA (Autonomous Financial Research Agent). "
            "All factual claims are sourced from the data sources listed above. "
            "This is not investment advice.*"
        )

        report_text = "\n".join(report_lines)

        return ToolResult(
            success=True,
            data={
                "report": report_text,
                "template": template_name,
                "sections_completed": sections_completed,
                "sections_total": len(template["sections"]),
                "sections_missing": sections_missing,
                "word_count": len(report_text.split()),
            },
            source=self.name,
            metadata={
                "template": template_name,
                "completeness": sections_completed / len(template["sections"]),
            },
        )
