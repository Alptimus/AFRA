"""
Agent Core — Main LangGraph workflow for the autonomous financial research agent.

Implements a Hybrid ReAct + Plan-and-Execute architecture:
1. ANALYZE: Parse and disambiguate the query
2. PLAN: Generate a structured research plan
3. EXECUTE: Execute each plan step using ReAct tool calls
4. SYNTHESIZE: Combine findings into a coherent narrative
5. VERIFY: Fact-check all claims
6. REPORT: Generate the final structured report
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Annotated, Any, TypedDict

from config.settings import Settings, get_settings
from agent.circuit_breaker import CircuitBreaker
from agent.disambiguation import QueryDisambiguator
from agent.error_handler import AgentError, ErrorCategory, ErrorHandler, ErrorSeverity
from agent.parser import ResponseParser
from agent.prompts import (
    DISAMBIGUATION_PROMPT,
    PLANNING_PROMPT,
    SELF_REFLECTION_PROMPT,
    SYNTHESIS_PROMPT,
    VERIFICATION_PROMPT,
    build_system_prompt,
)
from agent.query_analyzer import AnalyzedQuery, QueryAnalyzer
from tools.base import ToolResult
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── State Definition ──────────────────────────────────────────────────────────


class ResearchState(TypedDict, total=False):
    """Shared state for the LangGraph research workflow.

    All nodes read and write to this shared state object.
    """

    # Input
    query: str

    # Analysis
    analyzed_query: dict[str, Any]
    disambiguation: dict[str, Any]

    # Planning
    plan: list[dict[str, Any]]
    current_step: int

    # Execution
    gathered_data: dict[str, Any]
    tool_trace: list[dict[str, Any]]
    iteration_count: int

    # Synthesis & Verification
    synthesis_result: str
    verification_result: dict[str, Any]

    # Output
    final_report: str
    errors: list[dict[str, Any]]

    # Metadata
    start_time: float
    total_tool_calls: int


# ── Agent Class ───────────────────────────────────────────────────────────────


class FinancialResearchAgent:
    """Autonomous Financial Research Agent.

    Orchestrates the full research pipeline from query to report using
    a LangGraph-powered stateful workflow with 12 integrated tools.

    Architecture:
        Hybrid ReAct + Plan-and-Execute
        - Plan phase: LLM generates a structured research plan
        - Execute phase: ReAct loop executes each plan step with tool calls
        - Synthesize phase: Multi-source data synthesis
        - Verify phase: Fact-check all claims
        - Report phase: Generate final structured report

    Usage:
        agent = FinancialResearchAgent(settings, tool_registry)
        report = await agent.research("Prepare a risk assessment for Tesla Inc.")
        print(report)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._registry = tool_registry or ToolRegistry(self._settings)
        self._query_analyzer = QueryAnalyzer(self._settings)
        self._disambiguator = QueryDisambiguator(self._settings)
        self._error_handler = ErrorHandler(max_retries=self._settings.max_retries)
        self._circuit_breaker = CircuitBreaker()
        self._parser = ResponseParser()
        self._llm = None  # Initialized lazily

    # ── LLM Initialization ────────────────────────────────────────────────

    def _get_llm(self) -> Any:
        """Lazily initialize the LLM client based on provider setting."""
        if self._llm is not None:
            return self._llm

        provider = self._settings.llm_provider
        model = self._settings.llm_model

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=model,
                api_key=self._settings.openai_api_key,
                temperature=0.1,  # Low temperature for factual research
                max_tokens=4096,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model=model,
                api_key=self._settings.anthropic_api_key,
                temperature=0.1,
                max_tokens=4096,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        logger.info("Initialized LLM: %s/%s", provider, model)
        return self._llm

    # ── Main Research Entry Point ─────────────────────────────────────────

    async def research(self, query: str) -> str:
        """Execute the full research pipeline for a given query.

        Args:
            query: The financial research query/task.

        Returns:
            The final research report as a markdown string.
        """
        logger.info("Starting research for query: %s", query[:100])
        start_time = time.time()

        # Initialize state
        state: ResearchState = {
            "query": query,
            "analyzed_query": {},
            "disambiguation": {},
            "plan": [],
            "current_step": 0,
            "gathered_data": {},
            "tool_trace": [],
            "iteration_count": 0,
            "synthesis_result": "",
            "verification_result": {},
            "final_report": "",
            "errors": [],
            "start_time": start_time,
            "total_tool_calls": 0,
        }

        try:
            # Step 1: Analyze query
            state = await self._analyze_query(state)

            # Step 2: Plan research
            state = await self._plan_research(state)

            # Step 3: Execute research plan
            state = await self._execute_plan(state)

            # Step 4: Synthesize findings
            state = await self._synthesize(state)

            # Step 5: Verify facts
            state = await self._verify(state)

            # Step 6: Generate report
            state = await self._generate_report(state)

        except Exception as e:
            logger.error("Research pipeline error: %s", e, exc_info=True)
            error = self._error_handler.classify_error(e)
            self._error_handler.record_error(error)
            state["errors"].append({
                "phase": "pipeline",
                "error": str(e),
            })
            # Generate a partial report with what we have
            state = await self._generate_degraded_report(state)

        elapsed = time.time() - start_time
        logger.info(
            "Research completed in %.1fs, %d tool calls, %d errors",
            elapsed,
            state.get("total_tool_calls", 0),
            len(state.get("errors", [])),
        )

        return state.get("final_report", "Error: No report generated")

    # ── Pipeline Stages ───────────────────────────────────────────────────

    async def _analyze_query(self, state: ResearchState) -> ResearchState:
        """Stage 1: Analyze and disambiguate the query."""
        logger.info("Stage 1: Analyzing query")
        query = state["query"]

        # Rule-based analysis
        analysis = self._query_analyzer.analyze(query)
        state["analyzed_query"] = {
            "query_type": analysis.query_type,
            "complexity": analysis.complexity,
            "entities": analysis.entities,
            "time_period": analysis.time_period,
            "ambiguity_level": analysis.ambiguity_level,
            "ambiguities": analysis.ambiguities,
            "sub_queries": analysis.sub_queries,
            "recommended_tools": analysis.recommended_tools,
            "estimated_tool_calls": analysis.estimated_tool_calls,
        }

        # Disambiguation if needed
        if analysis.ambiguity_level in ("medium", "high"):
            disambiguation = self._disambiguator.disambiguate(
                query,
                analysis.ambiguity_level,
                analysis.ambiguities,
                analysis.entities,
            )
            state["disambiguation"] = {
                "assumptions": disambiguation.assumptions,
                "confidence": disambiguation.confidence,
                "edge_cases": disambiguation.edge_case_handling,
            }

        logger.info(
            "Query analysis: type=%s, complexity=%s, entities=%d, ambiguity=%s",
            analysis.query_type,
            analysis.complexity,
            len(analysis.entities),
            analysis.ambiguity_level,
        )
        return state

    async def _plan_research(self, state: ResearchState) -> ResearchState:
        """Stage 2: Generate a structured research plan using the LLM."""
        logger.info("Stage 2: Planning research")
        query = state["query"]
        analysis = state["analyzed_query"]

        try:
            llm = self._get_llm()
            prompt = PLANNING_PROMPT.format(
                query=query,
                query_analysis=json.dumps(analysis, indent=2, default=str),
            )

            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse the plan from LLM response
            plan = self._parse_plan(content)
            state["plan"] = plan
            logger.info("Research plan created with %d steps", len(plan))

        except Exception as e:
            logger.warning("LLM planning failed: %s, using default plan", e)
            state["plan"] = self._generate_default_plan(analysis)

        return state

    async def _execute_plan(self, state: ResearchState) -> ResearchState:
        """Stage 3: Execute the research plan step by step.

        For each plan step, executes the specified tool call and stores results.
        Implements the ReAct loop within each step.
        """
        logger.info("Stage 3: Executing research plan")
        plan = state["plan"]
        max_calls = self._settings.max_tool_calls

        for i, step in enumerate(plan):
            if state["total_tool_calls"] >= max_calls:
                logger.warning(
                    "Max tool calls reached (%d), stopping execution", max_calls
                )
                break

            step_num = step.get("step_number", i + 1)
            tool_name = step.get("tool", "")
            params = step.get("parameters", {})
            description = step.get("description", "")

            logger.info("Executing step %d: %s via %s", step_num, description, tool_name)

            # Check circuit breaker
            if not self._circuit_breaker.can_execute(tool_name):
                logger.warning("Circuit breaker OPEN for %s, skipping", tool_name)
                state["errors"].append({
                    "phase": "execution",
                    "step": step_num,
                    "error": f"Circuit breaker open for {tool_name}",
                })
                continue

            # Execute tool with fallback
            try:
                result = await self._registry.execute_with_fallback(tool_name, **params)
                state["total_tool_calls"] += 1

                if result.success:
                    self._circuit_breaker.record_success(tool_name)
                    state["gathered_data"][f"step_{step_num}_{tool_name}"] = {
                        "tool": result.source,
                        "data": result.data,
                        "metadata": result.metadata,
                        "execution_time_ms": result.execution_time_ms,
                    }
                else:
                    self._circuit_breaker.record_failure(tool_name)
                    error = AgentError(
                        category=ErrorCategory.TOOL_EXECUTION,
                        severity=ErrorSeverity.MEDIUM,
                        message=result.error,
                        tool_name=tool_name,
                    )
                    self._error_handler.record_error(error)
                    state["errors"].append({
                        "phase": "execution",
                        "step": step_num,
                        "tool": tool_name,
                        "error": result.error,
                    })

                # Record trace
                state["tool_trace"].append({
                    "step": step_num,
                    "tool": tool_name,
                    "params": params,
                    "success": result.success,
                    "time_ms": result.execution_time_ms,
                })

            except Exception as e:
                logger.error("Step %d execution error: %s", step_num, e)
                self._circuit_breaker.record_failure(tool_name)
                state["errors"].append({
                    "phase": "execution",
                    "step": step_num,
                    "tool": tool_name,
                    "error": str(e),
                })

            state["current_step"] = step_num

        logger.info(
            "Execution complete: %d steps, %d tool calls, %d data items",
            len(plan),
            state["total_tool_calls"],
            len(state["gathered_data"]),
        )
        return state

    async def _synthesize(self, state: ResearchState) -> ResearchState:
        """Stage 4: Synthesize gathered data into coherent analysis."""
        logger.info("Stage 4: Synthesizing findings")

        if not state["gathered_data"]:
            state["synthesis_result"] = "Insufficient data gathered for synthesis."
            return state

        try:
            llm = self._get_llm()

            # Prepare data summary for synthesis
            data_summary = self._summarize_gathered_data(state["gathered_data"])

            prompt = SYNTHESIS_PROMPT.format(
                query=state["query"],
                gathered_data=data_summary,
            )

            response = await llm.ainvoke(prompt)
            state["synthesis_result"] = (
                response.content if hasattr(response, "content") else str(response)
            )

        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            state["synthesis_result"] = self._fallback_synthesis(state["gathered_data"])

        return state

    async def _verify(self, state: ResearchState) -> ResearchState:
        """Stage 5: Verify factual claims in the synthesis."""
        logger.info("Stage 5: Verifying facts")

        if not state["synthesis_result"]:
            state["verification_result"] = {"skipped": True, "reason": "No synthesis to verify"}
            return state

        try:
            llm = self._get_llm()

            data_summary = self._summarize_gathered_data(state["gathered_data"])

            prompt = VERIFICATION_PROMPT.format(
                draft_report=state["synthesis_result"][:8000],
                source_data=data_summary[:8000],
            )

            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse as JSON
            try:
                state["verification_result"] = json.loads(content)
            except json.JSONDecodeError:
                state["verification_result"] = {
                    "raw_verification": content,
                    "parsed": False,
                }

        except Exception as e:
            logger.error("Verification failed: %s", e)
            state["verification_result"] = {
                "skipped": True,
                "reason": f"Verification error: {e}",
            }

        return state

    async def _generate_report(self, state: ResearchState) -> ResearchState:
        """Stage 6: Generate the final structured report."""
        logger.info("Stage 6: Generating final report")

        try:
            llm = self._get_llm()

            # Build the report using the LLM with all gathered context
            report_prompt = f"""Based on the following synthesized research findings, generate a
professional investment research report.

ORIGINAL QUERY: {state['query']}

SYNTHESIZED FINDINGS:
{state.get('synthesis_result', 'No synthesis available')}

VERIFICATION NOTES:
{json.dumps(state.get('verification_result', {}), indent=2, default=str)[:3000]}

QUERY ANALYSIS:
{json.dumps(state.get('analyzed_query', {}), indent=2, default=str)}

ASSUMPTIONS MADE:
{json.dumps(state.get('disambiguation', {}).get('assumptions', []), default=str)}

DEGRADATION NOTES:
{self._error_handler.get_degradation_report().get('degradation_note', 'None')}

FORMAT THE REPORT WITH THESE SECTIONS:
1. Executive Summary (3-5 key bullet points)
2. Company Overview
3. Financial Analysis (with data tables where possible)
4. Risk Assessment
5. Competitive Position
6. Research Methodology Notes (sources used, tools called, assumptions, confidence levels)

RULES:
- Cite sources inline: [Source: tool_name]
- Do NOT fabricate any data
- Note any sections that are incomplete due to data gaps
- Include the disclaimer at the end
"""
            response = await llm.ainvoke(report_prompt)
            report = response.content if hasattr(response, "content") else str(response)

            # Append methodology footer
            elapsed = time.time() - state.get("start_time", time.time())
            footer = f"""

---
## Research Metadata
- **Query**: {state['query']}
- **Total Tool Calls**: {state.get('total_tool_calls', 0)}
- **Data Sources Used**: {len(state.get('gathered_data', {}))}
- **Errors Encountered**: {len(state.get('errors', []))}
- **Recovery Rate**: {self._error_handler.recovery_rate:.0%}
- **Execution Time**: {elapsed:.1f}s
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

*This report was generated by AFRA (Autonomous Financial Research Agent).
This is not investment advice.*
"""
            state["final_report"] = report + footer

        except Exception as e:
            logger.error("Report generation failed: %s", e)
            state = await self._generate_degraded_report(state)

        return state

    async def _generate_degraded_report(self, state: ResearchState) -> ResearchState:
        """Generate a partial report when the full pipeline fails."""
        logger.info("Generating degraded report")
        degradation = self._error_handler.get_degradation_report()

        report_parts = [
            f"# Research Report (Partial — Degraded Output)",
            f"*Query: {state.get('query', 'Unknown')}*",
            "",
            "## ⚠️ Report Limitations",
            degradation.get("degradation_note", "Errors occurred during research."),
            "",
        ]

        # Include whatever data we do have
        if state.get("gathered_data"):
            report_parts.append("## Available Data")
            for key, value in state["gathered_data"].items():
                tool = value.get("tool", "unknown")
                report_parts.append(f"\n### Data from {tool}")
                data = value.get("data", {})
                if isinstance(data, dict):
                    for dk, dv in data.items():
                        if isinstance(dv, (str, int, float)):
                            report_parts.append(f"- **{dk}**: {dv}")

        report_parts.extend([
            "",
            "## Error Log",
            f"- Total errors: {degradation.get('total_errors', 0)}",
            f"- Recovered: {degradation.get('recovered_errors', 0)}",
            f"- Unrecovered: {degradation.get('unrecovered_errors', 0)}",
        ])

        state["final_report"] = "\n".join(report_parts)
        return state

    # ── Helper Methods ────────────────────────────────────────────────────

    def _parse_plan(self, content: str) -> list[dict[str, Any]]:
        """Parse the LLM's research plan output into structured steps."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: parse numbered steps
        steps = []
        lines = content.strip().split('\n')
        step_num = 1
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("Step")):
                steps.append({
                    "step_number": step_num,
                    "description": line,
                    "tool": "",
                    "parameters": {},
                })
                step_num += 1

        return steps if steps else self._generate_default_plan(
            {"recommended_tools": ["company_profile", "financial_data_api", "web_search"]}
        )

    def _generate_default_plan(
        self, analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate a default research plan when LLM planning fails."""
        entities = analysis.get("entities", [])
        ticker = ""
        for entity in entities:
            if entity.get("type") == "company" and entity.get("ticker"):
                ticker = entity["ticker"]
                break

        plan = [
            {
                "step_number": 1,
                "description": "Check long-term memory for existing research",
                "tool": "vector_db_search",
                "parameters": {"query": analysis.get("original_query", "financial research"), "top_k": 5},
            },
        ]

        if ticker:
            plan.extend([
                {
                    "step_number": 2,
                    "description": f"Get company profile for {ticker}",
                    "tool": "company_profile",
                    "parameters": {"ticker": ticker},
                },
                {
                    "step_number": 3,
                    "description": f"Get financial data for {ticker}",
                    "tool": "financial_data_api",
                    "parameters": {"ticker": ticker, "statement_type": "income_statement"},
                },
                {
                    "step_number": 4,
                    "description": f"Search for recent news about {ticker}",
                    "tool": "web_search",
                    "parameters": {"query": f"{ticker} latest news analysis 2024 2025"},
                },
                {
                    "step_number": 5,
                    "description": f"Analyze news sentiment for {ticker}",
                    "tool": "news_sentiment",
                    "parameters": {"query": ticker},
                },
            ])

        return plan

    def _summarize_gathered_data(
        self, gathered_data: dict[str, Any]
    ) -> str:
        """Create a text summary of gathered data for LLM context."""
        parts = []
        for key, value in gathered_data.items():
            tool = value.get("tool", "unknown")
            data = value.get("data", {})

            parts.append(f"--- Source: {tool} ---")

            if isinstance(data, dict):
                for dk, dv in data.items():
                    if isinstance(dv, (str, int, float, bool)):
                        parts.append(f"{dk}: {dv}")
                    elif isinstance(dv, list) and len(dv) <= 5:
                        parts.append(f"{dk}: {json.dumps(dv, default=str)[:500]}")
                    elif isinstance(dv, dict):
                        parts.append(f"{dk}: {json.dumps(dv, default=str)[:500]}")
            elif isinstance(data, str):
                parts.append(data[:1000])

            parts.append("")

        return "\n".join(parts)[:15000]  # Cap total context

    def _fallback_synthesis(self, gathered_data: dict[str, Any]) -> str:
        """Generate a simple synthesis when LLM synthesis fails."""
        parts = ["## Research Findings\n"]
        for key, value in gathered_data.items():
            tool = value.get("tool", "unknown")
            parts.append(f"### From {tool}")
            data = value.get("data", {})
            if isinstance(data, dict):
                for dk, dv in data.items():
                    if isinstance(dv, (str, int, float)):
                        parts.append(f"- **{dk}**: {dv}")
            parts.append("")
        return "\n".join(parts)

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def error_handler(self) -> ErrorHandler:
        """Access the error handler for metrics."""
        return self._error_handler

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Access the circuit breaker for status."""
        return self._circuit_breaker

    @property
    def tool_registry(self) -> ToolRegistry:
        """Access the tool registry."""
        return self._registry
