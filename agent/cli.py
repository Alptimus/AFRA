"""
AFRA CLI — Command-line interface for the Autonomous Financial Research Agent.

Usage:
    python -m agent.cli "Prepare a risk assessment for Tesla Inc."
    python -m agent.cli --query "Compare JPMorgan and Goldman Sachs"
    python -m agent.cli --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with Rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def create_agent():
    """Create and configure the research agent with all tools."""
    from config.settings import get_settings
    from tools.tool_registry import ToolRegistry, RateLimiter
    from tools.sec_edgar import SecFilingSearchTool
    from tools.web_search import WebSearchTool
    from tools.financial_api import FinancialDataTool
    from tools.news_sentiment import NewsSentimentTool
    from tools.earnings import EarningsTranscriptTool
    from tools.company_profile import CompanyProfileTool
    from tools.peer_comparison import PeerComparisonTool
    from tools.calculator import CalculationEngineTool
    from tools.fact_checker import FactCheckerTool
    from tools.report_gen import ReportGeneratorTool
    from tools.vector_search import VectorDBSearchTool
    from tools.vector_store import VectorDBStoreTool
    from memory.vector_store import VectorMemory
    from agent.core import FinancialResearchAgent

    settings = get_settings()
    registry = ToolRegistry(settings)
    vector_memory = VectorMemory(settings)

    # Register all 12 tools
    registry.register(SecFilingSearchTool(settings), RateLimiter(max_calls=10, window_seconds=1.0))
    registry.register(WebSearchTool(settings))
    registry.register(FinancialDataTool(settings))
    registry.register(NewsSentimentTool(settings))
    registry.register(EarningsTranscriptTool(settings))
    registry.register(CompanyProfileTool(settings))
    registry.register(PeerComparisonTool(settings))
    registry.register(CalculationEngineTool())
    registry.register(FactCheckerTool(settings, tool_registry=registry))
    registry.register(ReportGeneratorTool(settings))
    registry.register(VectorDBSearchTool(settings, vector_memory=vector_memory))
    registry.register(VectorDBStoreTool(settings, vector_memory=vector_memory))

    agent = FinancialResearchAgent(settings=settings, tool_registry=registry)

    console.print(f"[green]✓[/green] Agent initialized with {len(registry.list_tools())} tools")
    console.print(f"  Tools: {', '.join(registry.list_tools())}")

    return agent


async def run_research(query: str) -> None:
    """Run a research query and display the results."""
    console.print(Panel(
        f"[bold cyan]Research Query[/bold cyan]\n{query}",
        title="AFRA",
        border_style="cyan",
    ))
    console.print()

    start = time.time()

    with console.status("[bold green]Initializing agent..."):
        agent = create_agent()

    console.print(f"[bold green]Starting research...[/bold green]\n")

    report = await agent.research(query)
    elapsed = time.time() - start

    # Display the report
    console.print()
    console.print(Panel(
        Markdown(report),
        title="[bold green]Research Report[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Display stats
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")

    # Show tool usage stats
    stats = agent.tool_registry.get_tool_stats()
    if stats:
        console.print("\n[bold]Tool Usage Statistics:[/bold]")
        for name, s in stats.items():
            success_pct = s['success_rate'] * 100
            color = "green" if success_pct >= 80 else "yellow" if success_pct >= 50 else "red"
            console.print(
                f"  {name}: {s['call_count']} calls, "
                f"[{color}]{success_pct:.0f}% success[/{color}], "
                f"avg {s['avg_time_ms']:.0f}ms"
            )


async def interactive_mode() -> None:
    """Run the agent in interactive mode."""
    console.print(Panel(
        "[bold cyan]AFRA — Autonomous Financial Research Agent[/bold cyan]\n"
        "Interactive Mode. Type your research queries below.\n"
        "Type 'quit' or 'exit' to stop.",
        border_style="cyan",
    ))

    agent = create_agent()

    while True:
        console.print()
        query = console.input("[bold cyan]Research Query > [/bold cyan]")

        if query.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if not query.strip():
            continue

        report = await agent.research(query)
        console.print(Panel(Markdown(report), title="Report", border_style="green"))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AFRA — Autonomous Financial Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Prepare a risk assessment for Tesla Inc."
  %(prog)s --query "Compare JPMorgan and Goldman Sachs in investment banking"
  %(prog)s --interactive
  %(prog)s --list-tools
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query to execute",
    )
    parser.add_argument(
        "-q", "--query",
        dest="query_flag",
        help="Research query (alternative to positional argument)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available tools and exit",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.list_tools:
        agent = create_agent()
        for tool_name in agent.tool_registry.list_tools():
            tool = agent.tool_registry.get_tool(tool_name)
            console.print(f"  [cyan]{tool.name}[/cyan]: {tool.description[:80]}...")
        return

    if args.interactive:
        asyncio.run(interactive_mode())
        return

    query = args.query or args.query_flag
    if not query:
        parser.print_help()
        sys.exit(1)

    asyncio.run(run_research(query))


if __name__ == "__main__":
    main()
