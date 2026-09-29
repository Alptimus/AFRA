"""
Evaluation Dashboard — Rich terminal and HTML evaluation report generator.

Generates detailed evaluation reports with per-challenge scores, cross-challenge
comparison, category radar views, and historical performance tracking.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.metrics import EvaluationFramework, EvaluationReport, MetricResult

logger = logging.getLogger(__name__)


class EvaluationDashboard:
    """Generates evaluation reports from agent research output.

    Supports two output formats:
    1. Rich terminal output (using the `rich` library)
    2. Markdown file output (for results/ directory)

    Usage:
        dashboard = EvaluationDashboard()
        dashboard.evaluate_challenge(
            challenge_id=1,
            query="Prepare a company profile for Microsoft",
            report_text=agent_output,
            tool_trace=trace,
            gathered_data=data,
            errors=errors,
            execution_time=120.0,
        )
        dashboard.generate_summary()
    """

    def __init__(self, results_dir: str | Path = "results") -> None:
        self._framework = EvaluationFramework()
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._challenge_results: dict[int, EvaluationReport] = {}

    def evaluate_challenge(
        self,
        challenge_id: int,
        query: str,
        report_text: str,
        tool_trace: list[dict[str, Any]] | None = None,
        gathered_data: dict[str, Any] | None = None,
        errors: list[dict[str, Any]] | None = None,
        execution_time: float = 0.0,
        memory_hits: int = 0,
        total_api_calls: int = 0,
    ) -> EvaluationReport:
        """Evaluate a single challenge and store results.

        Args:
            challenge_id: Challenge number (1-8).
            query: The research query for this challenge.
            report_text: Agent-generated research report text.
            tool_trace: List of tool call records.
            gathered_data: Raw gathered data from tools.
            errors: List of errors encountered.
            execution_time: Total execution time in seconds.
            memory_hits: Number of vector DB cache hits.
            total_api_calls: Total external API calls made.

        Returns:
            EvaluationReport with all metric results.
        """
        evaluation = self._framework.evaluate(
            research_report=report_text,
            tool_trace=tool_trace,
            gathered_data=gathered_data,
            errors=errors,
            execution_time=execution_time,
            memory_hits=memory_hits,
            total_api_calls=total_api_calls,
        )

        self._challenge_results[challenge_id] = evaluation

        # Save individual challenge report
        self._save_challenge_report(challenge_id, query, report_text, evaluation)

        return evaluation

    def generate_summary(self) -> str:
        """Generate a cross-challenge evaluation summary.

        Returns:
            Markdown-formatted evaluation summary.
        """
        if not self._challenge_results:
            return "No challenges have been evaluated yet."

        lines = [
            "# AFRA — Evaluation Report",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "---",
            "",
            "## Overall Summary",
            "",
        ]

        # Overall scores table
        lines.append("| Challenge | Overall Score | FA | CO | AD | CS | AB |")
        lines.append("|---|---|---|---|---|---|---|")

        overall_scores = []
        for cid in sorted(self._challenge_results.keys()):
            report = self._challenge_results[cid]
            cat = report.category_scores
            line = (
                f"| Challenge {cid} | {report.overall_score:.0%} "
                f"| {cat.get('Factual Accuracy', 0):.0%} "
                f"| {cat.get('Completeness', 0):.0%} "
                f"| {cat.get('Analytical Depth', 0):.0%} "
                f"| {cat.get('Coherence', 0):.0%} "
                f"| {cat.get('Agent Behavior', 0):.0%} |"
            )
            lines.append(line)
            overall_scores.append(report.overall_score)

        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        lines.append(f"| **Average** | **{avg_score:.0%}** | | | | | |")
        lines.append("")

        # Category averages
        lines.append("## Category Averages")
        lines.append("")
        categories = ["Factual Accuracy", "Completeness", "Analytical Depth", "Coherence", "Agent Behavior"]
        for cat in categories:
            scores = [
                r.category_scores.get(cat, 0)
                for r in self._challenge_results.values()
            ]
            avg = sum(scores) / len(scores) if scores else 0
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            lines.append(f"- **{cat}**: {bar} {avg:.0%}")
        lines.append("")

        # Per-metric detail
        lines.append("## Per-Metric Analysis")
        lines.append("")
        lines.append("| Metric | Avg Value | Pass Rate | Target |")
        lines.append("|---|---|---|---|")

        # Collect metrics across challenges
        metric_values: dict[str, list[Any]] = {}
        metric_passes: dict[str, list[bool]] = {}
        metric_targets: dict[str, Any] = {}

        for report in self._challenge_results.values():
            for metric in report.metrics:
                if metric.metric_id not in metric_values:
                    metric_values[metric.metric_id] = []
                    metric_passes[metric.metric_id] = []
                    metric_targets[metric.metric_id] = metric.target
                metric_values[metric.metric_id].append(metric.value)
                metric_passes[metric.metric_id].append(metric.passed)

        for mid in sorted(metric_values.keys()):
            values = metric_values[mid]
            passes = metric_passes[mid]
            target = metric_targets[mid]
            avg_val = sum(v for v in values if isinstance(v, (int, float))) / len(values) if values else 0
            pass_rate = sum(1 for p in passes if p) / len(passes) if passes else 0
            lines.append(f"| {mid} | {avg_val:.2f} | {pass_rate:.0%} | {target} |")

        lines.append("")

        # Weakest metrics
        lines.append("## Areas for Improvement")
        lines.append("")
        weak_metrics = sorted(
            metric_passes.items(),
            key=lambda x: sum(1 for p in x[1] if p) / len(x[1]) if x[1] else 0,
        )[:5]
        for mid, passes in weak_metrics:
            pass_rate = sum(1 for p in passes if p) / len(passes) if passes else 0
            if pass_rate < 1.0:
                lines.append(f"- **{mid}**: Pass rate {pass_rate:.0%} — needs attention")

        summary = "\n".join(lines)

        # Save to file
        summary_path = self._results_dir / "evaluation_report.md"
        summary_path.write_text(summary)
        logger.info("Evaluation summary saved to %s", summary_path)

        return summary

    def _save_challenge_report(
        self,
        challenge_id: int,
        query: str,
        report_text: str,
        evaluation: EvaluationReport,
    ) -> None:
        """Save a single challenge's report and evaluation."""
        lines = [
            f"# Challenge {challenge_id} — Results",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            f"**Query:** {query}",
            f"**Overall Score:** {evaluation.overall_score:.0%}",
            "",
            "---",
            "",
            "## Research Report",
            "",
            report_text,
            "",
            "---",
            "",
            "## Evaluation Metrics",
            "",
            self._framework.format_report(evaluation),
        ]

        path = self._results_dir / f"challenge_{challenge_id}.md"
        path.write_text("\n".join(lines))
        logger.info("Challenge %d report saved to %s", challenge_id, path)

    def print_summary(self) -> None:
        """Print evaluation summary to terminal using Rich."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console()

            # Header
            console.print(Panel(
                "[bold cyan]AFRA Evaluation Dashboard[/bold cyan]",
                border_style="cyan",
            ))

            # Scores table
            table = Table(title="Challenge Scores")
            table.add_column("Challenge", style="cyan")
            table.add_column("Overall", style="bold")
            table.add_column("Factual Acc.", style="green")
            table.add_column("Completeness", style="green")
            table.add_column("Anal. Depth", style="green")
            table.add_column("Coherence", style="green")
            table.add_column("Agent Behav.", style="green")

            for cid in sorted(self._challenge_results.keys()):
                report = self._challenge_results[cid]
                cat = report.category_scores
                table.add_row(
                    f"Challenge {cid}",
                    f"{report.overall_score:.0%}",
                    f"{cat.get('Factual Accuracy', 0):.0%}",
                    f"{cat.get('Completeness', 0):.0%}",
                    f"{cat.get('Analytical Depth', 0):.0%}",
                    f"{cat.get('Coherence', 0):.0%}",
                    f"{cat.get('Agent Behavior', 0):.0%}",
                )

            console.print(table)

        except ImportError:
            print(self.generate_summary())
