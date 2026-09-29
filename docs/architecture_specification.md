# AFRA — Architecture Specification

**Version:** 1.0
**Date:** September 2026
**Project:** Autonomous Financial Research Agent with Multi-Source Synthesis
**Code:** Project 1A

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Agent Architecture Pattern](#2-agent-architecture-pattern)
3. [Cognitive Loop Design](#3-cognitive-loop-design)
4. [Tool Registry Architecture](#4-tool-registry-architecture)
5. [Memory Architecture](#5-memory-architecture)
6. [RAG Pipeline Design](#6-rag-pipeline-design)
7. [Multi-Source Synthesis Engine](#7-multi-source-synthesis-engine)
8. [Error Handling & Resilience](#8-error-handling--resilience)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Data Flow & Sequence Diagrams](#10-data-flow--sequence-diagrams)
11. [Security & Configuration](#11-security--configuration)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)

---

## 1. Executive Summary

AFRA is an autonomous AI agent designed to perform investment research at a quality
level comparable to a junior financial analyst. Given a natural language query
(e.g., "Prepare a risk assessment for Tesla Inc."), the agent autonomously:

1. Analyzes and disambiguates the query
2. Plans a multi-step research strategy
3. Executes the plan by calling 12 specialized financial tools
4. Synthesizes findings from multiple sources with conflict resolution
5. Verifies all factual claims via cross-source checking
6. Produces a structured, citation-backed research report

The system is built on a **Hybrid ReAct + Plan-and-Execute** architecture using
LangGraph for stateful workflow orchestration. It integrates a 3-tier memory
system (short-term, long-term, episodic), a multi-source synthesis engine with
5-tier source hierarchy, and a comprehensive 22-metric evaluation framework.

### Key Design Principles

- **No hallucination**: Every factual claim must be traceable to a retrieved source
- **Transparent failure**: When tools fail, report the gap rather than fabricate data
- **Source hierarchy**: Audited filings > financial APIs > transcripts > news > social
- **Efficient tool use**: Check memory before external APIs; cache results aggressively
- **Modular architecture**: Each component (tools, memory, synthesis, evaluation)
  is independently testable and replaceable

---

## 2. Agent Architecture Pattern

### 2.1 Pattern Selection: Hybrid ReAct + Plan-and-Execute

We chose a **hybrid approach** combining the strengths of two established agent
patterns:

**ReAct (Reasoning + Acting):**
- Interleaves thought and action in a tight loop
- Strength: Handles unexpected tool outputs and adapts mid-stream
- Weakness: Can be inefficient with many sequential tool calls; no upfront planning

**Plan-and-Execute:**
- Creates a structured plan first, then executes each step
- Strength: Efficient for multi-step research with clear dependencies
- Weakness: Rigid; cannot easily adapt when plan assumptions break

**Our Hybrid Approach:**
- **Phase 1 (Plan-and-Execute):** The LLM creates a numbered research plan
  identifying what data to gather, which tools to use, and dependencies between steps.
- **Phase 2 (ReAct):** Each plan step is executed using a ReAct-style tool call loop
  that allows mid-step adaptation if tool results are unexpected.
- **Phase 3 (Synthesis):** A non-interactive synthesis phase combines findings.
- **Phase 4 (Verify):** Chain-of-Verification pattern checks all claims.
- **Phase 5 (Report):** Template-based report generation.

### 2.2 Why Not Pure ReAct?

Pure ReAct for financial research results in:
- Redundant API calls (no upfront planning = repeated data fetches)
- Poor tool ordering (may attempt synthesis before gathering sufficient data)
- No research strategy memory (cannot leverage past session learnings)

### 2.3 Why Not Pure Plan-and-Execute?

Pure planning fails when:
- A tool returns unexpected data requiring research pivot
- An API is down and fallback chains must be activated
- The initial query is ambiguous and needs mid-stream disambiguation

### 2.4 State Management

The agent uses a `ResearchState` TypedDict as the central state object
passed through the LangGraph workflow:

```python
class ResearchState(TypedDict, total=False):
    query: str                         # Original user query
    analyzed_query: dict[str, Any]     # Query type, entities, ambiguity
    disambiguation: dict[str, Any]     # Assumptions, edge cases
    plan: list[dict[str, Any]]         # Numbered research steps
    current_step: int                  # Execution progress
    gathered_data: dict[str, Any]      # Per-step tool results
    tool_trace: list[dict[str, Any]]   # Full execution trace
    iteration_count: int               # Safety counter
    synthesis_result: str              # Synthesized analysis
    verification_result: dict          # Fact-check results
    final_report: str                  # Final output
    errors: list[dict[str, Any]]       # Error log
    start_time: float                  # Timing
    total_tool_calls: int              # Budget tracking
```

---

## 3. Cognitive Loop Design

### 3.1 Six-Stage Pipeline

```
Query → [ANALYZE] → [PLAN] → [EXECUTE] → [SYNTHESIZE] → [VERIFY] → [REPORT]
             │          │         │             │            │           │
         Query       Research  ReAct        Multi-src    Chain-of     Template
         Analyzer    Planner   Tool Loop    Synthesis    Verification  Gen
```

**Stage 1 — ANALYZE (Query Analyzer + Disambiguator)**
- Entity extraction: Identifies companies, tickers, sectors from natural language
- Type classification: factual / analytical / comparative / risk_assessment / sector_analysis
- Complexity assessment: simple (4 calls) / moderate (8) / complex (15)
- Ambiguity detection: Identifies unclear references, missing time periods
- Edge case flagging: Private companies, new IPOs, M&A situations

**Stage 2 — PLAN (LLM-Powered Research Planning)**
- Input: Analyzed query, available tools, past episodic memory
- Output: Numbered plan steps with tool assignments and parameters
- Always starts with `vector_db_search` (check memory before external calls)
- Fallback: If LLM planning fails, a rule-based default plan is generated

**Stage 3 — EXECUTE (ReAct Tool Execution)**
- Iterates through each plan step
- For each step: checks circuit breaker → checks rate limiter → executes tool →
  records result → updates state
- Falls back through the tool's fallback chain on failure
- Stops at `max_tool_calls` (default 20) to prevent runaway execution

**Stage 4 — SYNTHESIZE (Multi-Source Synthesis Engine)**
- Categorizes gathered data by source type
- Detects conflicts across sources
- Resolves conflicts using the 5-tier source hierarchy
- Generates narrative threads connecting cross-source findings
- Identifies sentiment-fact alignment or misalignment

**Stage 5 — VERIFY (Fact-Checking)**
- Identifies all factual claims in the synthesis
- Cross-references each claim against source data
- Flags unsourced claims, contradictions, temporal errors, entity errors
- Computes overall accuracy estimate

**Stage 6 — REPORT (Template-Based Generation)**
- Formats verified synthesis into structured markdown
- Ensures all required sections are present
- Injects methodology notes, source attribution, and disclaimers
- Appends execution metadata (tool calls, time, errors, recovery rate)

### 3.2 Safety Mechanisms

- **Tool call budget**: Maximum 20 calls per research task
- **Iteration counter**: Prevents infinite loops in the execution phase
- **Circuit breaker**: Opens after 3 consecutive failures on a single tool
- **Timeout**: Individual tool calls timeout at 30-60 seconds

---

## 4. Tool Registry Architecture

### 4.1 Registry Design

The `ToolRegistry` serves as the central catalog for all 12 agent tools:

```
ToolRegistry
├── Registration: Maps tool name → BaseTool instance
├── Discovery: Exports schemas in OpenAI or Anthropic format
├── Execution: Routes tool calls with rate limiting + caching
├── Fallback: Chains alternative tools on failure
└── Telemetry: Tracks per-tool call counts, success rates, timing
```

### 4.2 Tool Categories

| Category | Tools | Purpose |
|---|---|---|
| Data Retrieval | sec_filing_search, financial_data_api, earnings_transcript, company_profile | Primary data sources |
| Information Gathering | web_search, news_sentiment, peer_comparison | Supplementary intelligence |
| Internal / Analytical | calculation_engine, fact_checker, report_generator | Processing tools |
| Memory | vector_db_search, vector_db_store | Long-term memory interface |

### 4.3 Schema Design

Each tool defines a JSON Schema following the OpenAI function-calling specification.
Schemas are stored both as Python class attributes (`parameters_schema`) and as
standalone JSON files (`tools/schemas/*.json`) for external tooling compatibility.

Schema format:
```json
{
  "name": "tool_name",
  "description": "When and why to use this tool (1-3 sentences)",
  "parameters": {
    "type": "object",
    "properties": { ... },
    "required": [ ... ]
  }
}
```

### 4.4 Caching Strategy

Tool results are cached using a content-addressable hash of `(tool_name, kwargs)`.
Cache entries have a default TTL of 300 seconds (5 minutes). Financial data that
changes frequently (stock prices, news) uses shorter TTLs; SEC filings use longer
TTLs (filings don't change after publication).

### 4.5 Rate Limiting

Token-bucket rate limiter per tool:
- SEC EDGAR: 10 requests/second (SEC policy)
- Financial APIs: depends on API tier
- Web search: 10 requests/minute (Tavily free tier)

---

## 5. Memory Architecture

### 5.1 Three-Tier Memory System

```
┌─────────────────────────────────────────────┐
│              Short-Term Memory               │
│        (ContextManager — in-session)         │
│  Token budget: 40% primary / 30% supporting  │
│  Progressive summarization of old context     │
├─────────────────────────────────────────────┤
│              Long-Term Memory                │
│        (VectorMemory — Chroma DB)            │
│  Semantic search across past research         │
│  Intelligent document chunking               │
│  Metadata filtering (ticker, source, date)    │
├─────────────────────────────────────────────┤
│              Episodic Memory                 │
│        (EpisodicMemory — JSON files)         │
│  Records research session outcomes            │
│  Learns which strategies work best            │
│  Informs future research planning             │
└─────────────────────────────────────────────┘
```

### 5.2 Short-Term Memory (Context Manager)

**Purpose:** Manage the agent's working memory within LLM context window limits.

**Token Budget Allocation:**
- 40% — Primary data (SEC filings, financial statements)
- 30% — Supporting data (news, analyst reports, sentiment)
- 20% — System prompt and tool descriptions
- 10% — Reserved for LLM output generation

**Compression Strategy:**
When approaching token limits, the context manager:
1. Ranks observations by importance score (0-1)
2. Keeps the top 50% highest-importance observations intact
3. Compresses the remaining observations into a bullet-point summary
4. Prepends the summary to the context as "Previous Research Summary"

### 5.3 Long-Term Memory (Vector Store)

**Backend:** Chroma (development) with Pinecone/Qdrant as production options.

**Document Schema:**
- `id`: Deterministic hash-based ID (content + ticker + source)
- `content`: The text chunk
- `embedding`: Vector embedding (text-embedding-3-small, 1536 dims)
- `ticker`: Company ticker symbol
- `source_type`: Origin of data (10-K, earnings_call, news, etc.)
- `date`: Date of the original data
- `confidence`: Data reliability score (0-1)
- `stored_at`: When the data was cached

**Chunking Strategies:**
- SEC Filings → Section-based (Risk Factors, MD&A, financial statements)
- Earnings Transcripts → Speaker-turn based (preserves Q&A context)
- News Articles → Paragraph-based with headline context overlap
- Financial Data → Stored as structured JSON with metadata filtering

### 5.4 Episodic Memory (Strategy Learning)

**Purpose:** Learn from past research sessions to improve future planning.

Records per session:
- Query type and entities
- Research strategy (tool order and parameters)
- Per-tool success rates and timing
- Errors encountered and recovery actions
- Overall outcome quality

**Usage:** Before planning a new research task, the agent checks episodic
memory for past sessions of the same query type. If relevant episodes exist,
the recommended tools and execution order are incorporated into the plan.

---

## 6. RAG Pipeline Design

### 6.1 Retrieval-Augmented Generation Flow

```
Query → Query Analyzer → Sub-Query Decomposition
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Vector Search    External APIs    Web Search
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    Context Assembly
                    (Token Budgeting)
                              │
                              ▼
                    LLM Generation
                    (with citations)
```

### 6.2 Sub-Query Decomposition

For complex queries, the `QueryAnalyzer` generates multiple sub-queries
optimized for different retrieval tools:

Example for "Analyze Tesla's financial performance and competitive position":
1. `"Tesla TSLA financial performance revenue margins"` → financial_data_api
2. `"Tesla TSLA recent news developments"` → web_search
3. `"Tesla TSLA competitive position market share"` → peer_comparison
4. `"Tesla TSLA growth strategy outlook"` → earnings_transcript

### 6.3 Citation Enforcement

The system prompt mandates inline citations in the format:
`[Source: tool_name, date]`

The evaluation framework's FA-2 metric checks citation accuracy,
and the VERIFY stage flags any claim without a supporting source.

---

## 7. Multi-Source Synthesis Engine

### 7.1 Source Reliability Hierarchy (CORRECTED)

```
Tier 1 (Highest): SEC Filings (10-K, 10-Q) — legally mandated, audited
Tier 2:           Financial Data APIs — professionally curated
Tier 3:           Earnings Transcripts — direct management commentary
Tier 4:           Professional News Outlets — editorial oversight
Tier 5 (Lowest):  Social Media / Forums — unverified
```

> **Note:** The project documentation (Section A6.2) incorrectly placed social
> media at Tier 4 and news at Tier 5. This implementation uses the corrected
> hierarchy. See ERROR_LOG.md #1.

### 7.2 Conflict Resolution Protocol

When the same data point appears with different values across sources:

1. **Identify** the conflicting metric and all reported values
2. **Assess** the tier of each source
3. **Check temporal** differences (different reporting periods?)
4. **Check for restatements** (<1% difference = rounding)
5. **Apply highest-tier rule** — the highest-tier (lowest number) source wins
6. **Document** the conflict in the methodology notes

### 7.3 Synthesis Techniques

1. **Narrative Threading** — Connects data points from different sources into
   thematic story arcs (growth trajectory, risk factors, market sentiment)

2. **Quantitative Triangulation** — Requires numerical claims to be confirmed
   by at least 2 independent sources before inclusion

3. **Sentiment-Fact Alignment** — Compares qualitative signals (news tone,
   management language) with quantitative data (revenue growth, margins).
   Misalignment is flagged as a key analytical finding.

---

## 8. Error Handling & Resilience

### 8.1 Error Categories

| Category | Examples | Severity Range |
|---|---|---|
| Tool Execution | API timeouts, rate limits, HTTP errors | Medium |
| Reasoning | Hallucination, logical errors | High |
| Data Quality | Stale data, conflicts, misattribution | Low–Medium |
| Configuration | Missing API keys, bad config | High |
| Network | DNS failures, SSL errors, connection timeouts | Medium |

### 8.2 Retry Strategy

Exponential backoff with jitter:
- Base delay: 1 second
- Multiplier: 2× per attempt
- Jitter: 0-500ms random per attempt
- Maximum retries: 5 (configurable)
- Sequence: 1s, 2s, 4s, 8s, 16s (+ random jitter)

### 8.3 Circuit Breaker

Prevents cascading failures when a tool is consistently failing:

```
CLOSED (normal) ──[3 consecutive failures]──→ OPEN (blocking)
                                                     │
                                              [60s timeout]
                                                     │
                                                     ▼
                                               HALF_OPEN (testing)
                                                 │         │
                                            [success]  [failure]
                                                 │         │
                                                 ▼         ▼
                                              CLOSED     OPEN
```

### 8.4 Fallback Chains

Every external tool has a defined fallback chain:
```
sec_filing_search   → web_search → vector_db_search
financial_data_api  → sec_filing_search → web_search → vector_db_search
web_search          → news_sentiment → vector_db_search
earnings_transcript → web_search → vector_db_search
news_sentiment      → web_search
company_profile     → web_search → vector_db_search
peer_comparison     → web_search → calculation_engine
```

### 8.5 Graceful Degradation Protocol

When a tool fails and all fallbacks are exhausted:
1. The error is recorded with category and severity
2. The research continues with available data
3. The final report explicitly states which sections are incomplete and why
4. The methodology notes list which tools were unavailable
5. **No data is fabricated** to fill gaps

---

## 9. Evaluation Framework

### 9.1 Metric Categories and Weights

| Category | Weight | # Metrics | Focus |
|---|---|---|---|
| Factual Accuracy | 30% | 5 (FA-1..FA-5) | Correctness of all claims |
| Completeness | 20% | 4 (CO-1..CO-4) | Section and source coverage |
| Analytical Depth | 20% | 4 (AD-1..AD-4) | Insight quality and reasoning |
| Coherence | 15% | 4 (CS-1..CS-4) | Structure and professional quality |
| Agent Behavior | 15% | 5 (AB-1..AB-5) | Efficiency and resilience |

### 9.2 Critical Metric: AB-4 Memory Utilization (CORRECTED)

The project documentation states AB-4 is "calculated as memory_hits **multiplied
by** total_api_calls." This produces a meaningless large number.

**Correct formula:** `AB-4 = memory_hits / total_api_calls`

This produces a ratio between 0 and 1, consistent with the target (≥0.3) and
the metric's stated purpose ("ratio of memory hits to total external API calls").
See ERROR_LOG.md #2.

---

## 10. Data Flow & Sequence Diagrams

### 10.1 End-to-End Research Flow

```
User Query
    │
    ▼
QueryAnalyzer.analyze()
    │ → entities, query_type, complexity, ambiguity
    ▼
QueryDisambiguator.disambiguate() [if ambiguity ≥ medium]
    │ → assumptions, edge_case_handling
    ▼
LLM → Research Plan (numbered steps with tools + params)
    │
    ▼
┌─ For each plan step: ─────────────────────────┐
│  CircuitBreaker.can_execute(tool)?             │
│     NO → skip + log error                      │
│     YES → RateLimiter.acquire()?               │
│        NO → wait or skip                       │
│        YES → Cache.check(tool, params)?         │
│           HIT → return cached                   │
│           MISS → tool.execute(**params)          │
│              SUCCESS → cache + record trace      │
│              FAILURE → try fallback chain         │
│                 ALL FAILED → record error + skip │
└────────────────────────────────────────────────┘
    │
    ▼
SynthesisEngine.synthesize(gathered_data)
    │ → narrative, conflicts, insights, confidence
    ▼
LLM → Verification (Chain-of-Verification)
    │ → flagged claims, accuracy estimate
    ▼
LLM → Final Report Generation
    │ → structured markdown with citations
    ▼
Report + Metadata Footer
```

### 10.2 Tool Execution Sequence

```
Registry.execute(tool_name, **kwargs)
    │
    ├── Check cache → HIT? return cached result
    │
    ├── Check rate limiter → BLOCKED? return rate_limit error
    │
    ├── tool.validate_inputs(**kwargs)
    │       FAIL → return ValidationError
    │
    ├── tool._execute(**kwargs)
    │       │
    │       ├── SUCCESS → wrap in ToolResult(success=True, data=...)
    │       │               cache result, log call, return
    │       │
    │       └── FAILURE → try fallback chain
    │                       │
    │                       ├── Fallback succeeds → return result + metadata
    │                       └── All fail → return ToolResult(success=False)
    │
    └── Log ToolCallRecord (timing, success, tool_name)
```

---

## 11. Security & Configuration

### 11.1 Secret Management

- All API keys loaded from environment variables via `.env` file
- Pydantic `BaseSettings` validates and types all configuration
- `.env` is gitignored; `.env.example` provides the template
- No secrets are ever logged or included in research output

### 11.2 SEC EDGAR Compliance

- User-Agent header required on all EDGAR requests
- Rate limit compliance: ≤10 requests/second
- No automated bulk downloading of filings

### 11.3 Input Sanitization

- Custom formula evaluation in `CalculationEngineTool._calc_custom()` uses
  regex whitelist (`^[\d\s\+\-\*/\(\)\.\,a-zA-Z_]+$`) and restricted
  `eval()` with `__builtins__` stripped
- Tool inputs validated against JSON Schema before execution

---

## 12. Design Decisions & Trade-offs

### 12.1 LangGraph vs. Custom Orchestration

**Decision:** Use LangGraph for the high-level pipeline, but custom Python for
individual tool orchestration within the execution phase.

**Rationale:** LangGraph provides state persistence, checkpointing, and visual
debugging for the 6-stage pipeline. However, the ReAct tool-calling loop within
the EXECUTE stage is simpler to implement and debug as a standard Python async loop.

### 12.2 Chroma vs. Pinecone/Qdrant

**Decision:** Chroma for development, with adapter pattern for production alternatives.

**Rationale:** Chroma requires zero infrastructure (embedded, file-based). For
production with >100K documents, Pinecone or Qdrant would provide better
performance. The `VectorMemory` class abstracts the backend.

### 12.3 Synchronous vs. Asynchronous

**Decision:** Fully async architecture using `asyncio` and `httpx.AsyncClient`.

**Rationale:** Financial data APIs have high latency (200-2000ms). Async
execution allows parallel tool calls where plan steps are independent,
reducing total research time.

### 12.4 LLM Temperature

**Decision:** Temperature 0.1 for all research tasks.

**Rationale:** Financial research demands factual accuracy over creativity.
Low temperature minimizes hallucination risk while still allowing flexible
language generation.

### 12.5 Embedding Model

**Decision:** `text-embedding-3-small` (1536 dimensions) for development.

**Rationale:** Balances cost ($0.02/M tokens) with quality. The 3072-dimension
`text-embedding-3-large` provides marginal quality improvement but 6.5× cost.
For financial text where domain-specific terms are well-represented in the
training data, the smaller model is sufficient.

> **Note:** The project documentation (Section E2.2) incorrectly states
> text-embedding-3-large has 1024 dimensions. The actual default is 3072.
> See ERROR_LOG.md #3.

---

*End of Architecture Specification*
