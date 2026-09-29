# AFRA — Autonomous Financial Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AFRA** is a hybrid ReAct + Plan-and-Execute autonomous agent that produces
investment-quality financial research reports from a single natural language query.
It orchestrates 12 specialized tools, a 3-tier memory system, multi-source synthesis
with conflict resolution, and a 22-metric evaluation framework.

---

## ✨ Key Features

| Capability | Implementation |
|---|---|
| **12 Tools** | SEC EDGAR, Financial APIs, Web Search, Earnings Transcripts, News Sentiment, Company Profiles, Peer Comparison, Calculator, Fact Checker, Report Generator, Vector Search/Store |
| **Hybrid Architecture** | Plan-and-Execute for strategy + ReAct for tool execution |
| **3-Tier Memory** | Short-term (context window), Long-term (Chroma vector DB), Episodic (strategy learning) |
| **Multi-Source Synthesis** | Narrative threading, quantitative triangulation, sentiment-fact alignment |
| **Conflict Resolution** | 5-tier source reliability hierarchy with 6-step resolution protocol |
| **Error Resilience** | Exponential backoff retry, circuit breakers, fallback chains, graceful degradation |
| **22-Metric Evaluation** | Factual accuracy, completeness, analytical depth, coherence, agent behavior |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│                   CLI / API                  │
│               (agent/cli.py)                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│          Agent Core (LangGraph)              │
│  ┌──────────┬──────────┬──────────┐        │
│  │ Analyze  │   Plan   │ Execute  │        │
│  │ (Query   │ (LLM     │ (ReAct   │        │
│  │ Analyzer)│  Plan)   │  Loop)   │        │
│  └──────────┴──────────┴──────────┘        │
│  ┌──────────┬──────────┬──────────┐        │
│  │Synthesize│  Verify  │  Report  │        │
│  │(Multi-src│ (Fact    │(Template │        │
│  │ Engine)  │  Check)  │  Gen)    │        │
│  └──────────┴──────────┴──────────┘        │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────┼─────────────┐
    ▼            ▼             ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│ Tools  │ │  Memory  │ │Evaluation│
│Registry│ │  System  │ │Framework │
│(12)    │ │(3-tier)  │ │(22 KPIs) │
└────────┘ └──────────┘ └──────────┘
```

---

## 📦 Project Structure

```
AFRA/
├── agent/                      # Core agent logic
│   ├── core.py                 # LangGraph 6-stage pipeline
│   ├── prompts.py              # System & specialized prompts
│   ├── parser.py               # LLM response parser
│   ├── query_analyzer.py       # Query classification & decomposition
│   ├── disambiguation.py       # Ambiguity resolution
│   ├── error_handler.py        # Retry, classification, degradation
│   ├── circuit_breaker.py      # Cascading failure prevention
│   ├── fallback_chains.py      # Tool fallback definitions
│   └── cli.py                  # Command-line interface
├── tools/                      # 12 tool implementations
│   ├── base.py                 # BaseTool ABC + ToolResult
│   ├── tool_registry.py        # Central catalog with caching/rate-limiting
│   ├── sec_edgar.py            # SEC EDGAR filing retrieval
│   ├── web_search.py           # Tavily web search
│   ├── financial_api.py        # FMP / yfinance financial data
│   ├── news_sentiment.py       # NewsAPI + TextBlob sentiment
│   ├── earnings.py             # Earnings call transcripts
│   ├── company_profile.py      # Company overview data
│   ├── peer_comparison.py      # Peer identification & comparison
│   ├── calculator.py           # DCF, ratios, CAGR, WACC, stats
│   ├── fact_checker.py         # Cross-source verification
│   ├── report_gen.py           # Template-based report generation
│   ├── vector_search.py        # Long-term memory search
│   └── vector_store.py         # Long-term memory storage
├── memory/                     # 3-tier memory system
│   ├── vector_store.py         # Chroma vector DB with smart chunking
│   ├── context_manager.py      # Short-term token-budgeted context
│   └── episodic.py             # Strategy learning from past sessions
├── synthesis/                  # Multi-source synthesis
│   ├── engine.py               # Synthesis orchestrator
│   ├── conflict_resolver.py    # 6-step conflict resolution
│   └── narrative.py            # Narrative threading
├── evaluation/                 # Quality assessment
│   └── metrics.py              # 22 KPI metrics framework
├── config/                     # Configuration
│   └── settings.py             # Pydantic BaseSettings
├── tests/                      # Test suite
│   └── test_tools.py           # Tool & registry tests
├── docs/                       # Documentation (TBD)
├── data/                       # Runtime data (gitignored)
├── ERROR_LOG.md                # Deliberate error documentation
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Modern Python project config
├── .env.example                # API key template
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd AFRA
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY (required for LLM)
# - TAVILY_API_KEY (recommended for web search)
# - FMP_API_KEY (recommended for financial data)
```

### 3. Run a Research Query

```bash
# Single query
python -m agent.cli "Prepare a risk assessment for Tesla Inc."

# Interactive mode
python -m agent.cli --interactive

# List available tools
python -m agent.cli --list-tools
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 🔧 Configuration

All configuration is managed through environment variables and the `.env` file:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes* | OpenAI API key for LLM and embeddings |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key (alternative to OpenAI) |
| `LLM_PROVIDER` | No | `openai` (default) or `anthropic` |
| `LLM_MODEL` | No | Model name (default: `gpt-4o-mini`) |
| `TAVILY_API_KEY` | Recommended | Tavily search API key |
| `FMP_API_KEY` | Recommended | Financial Modeling Prep API key |
| `NEWSAPI_KEY` | Optional | NewsAPI.org key for news sentiment |

\* One of OpenAI or Anthropic API key is required.

---

## 📊 Evaluation Framework

The agent is assessed on **22 metrics** across 5 categories:

| Category | Weight | Metrics |
|---|---|---|
| Factual Accuracy | 30% | FA-1 to FA-5 (numerical accuracy, citations, temporal, entity, hallucination) |
| Completeness | 20% | CO-1 to CO-4 (sections, source diversity, temporal coverage, risk factors) |
| Analytical Depth | 20% | AD-1 to AD-4 (insight density, cross-source, quantitative reasoning, forward-looking) |
| Coherence | 15% | CS-1 to CS-4 (logical flow, consistency, executive summary, formatting) |
| Agent Behavior | 15% | AB-1 to AB-5 (tool efficiency, error recovery, planning, memory utilization, latency) |

---

## 📄 License

[MIT License](LICENSE)