# ERROR_LOG.md — Deliberate Errors Found in Project Documentation

This document catalogs the **7 deliberate factual or logical errors** embedded
throughout Parts A through E of the AFRA project documentation, as required
by the project specification (Section A4.3, Error Detection Exercise).

---

## Error #1: Source Reliability Hierarchy — Incorrect Tier Ordering

**Location:** Section A6.2 (Part A, Page 21)

**The Error:** The document places **Social media posts and anonymous forum
discussions** at **Tier 4** and **Major news outlets (Reuters, Bloomberg News,
Financial Times)** at **Tier 5** (lowest reliability). This ranking is incorrect.

**Why It's Wrong:** Professional journalism from major news outlets (Reuters,
Bloomberg, Financial Times) has editorial oversight, fact-checking processes,
and legal accountability. Social media and anonymous forums are unverified,
subject to manipulation (pump-and-dump schemes, bot activity), and lack
editorial controls. News outlets should rank **higher** than social media.

**Correct Hierarchy:**
- Tier 4: Major news outlets (professional journalism with editorial oversight)
- Tier 5: Social media and anonymous forums (unverified, subject to manipulation)

**Implementation:** The `synthesis/conflict_resolver.py` module uses the
**corrected** hierarchy in its `SOURCE_TIERS` mapping.

---

## Error #2: AB-4 Memory Utilization Formula — Multiplication Instead of Division

**Location:** Section A5.2, Category 5 (Part A, Page 19)

**The Error:** The document states that metric AB-4 (Memory Utilization) is
*"calculated as memory_hits **multiplied by** total_api_calls"*.

**Why It's Wrong:** A utilization ratio should measure what fraction of total
calls were served from memory. Multiplying memory_hits by total_api_calls
produces a meaningless large number that grows as the agent makes more calls,
rather than measuring efficiency. The formula should be a **division** (ratio):

- **Wrong:** `memory_hits × total_api_calls`
- **Correct:** `memory_hits / total_api_calls`

This matches the text's own description ("ratio of memory hits to total
external API calls") and the stated target (≥0.3), which only makes sense
as a proportion between 0 and 1.

**Implementation:** The `evaluation/metrics.py` module uses the **corrected**
formula: `memory_hits / total_api_calls`.

---

## Error #3: text-embedding-3-large Dimensions — Wrong Dimension Count

**Location:** Section E2.2 (Part E, Page 62)

**The Error:** The document states that **text-embedding-3-large** has
**1024 dimensions**.

**Why It's Wrong:** OpenAI's text-embedding-3-large model produces embeddings
with **3072 dimensions** by default (not 1024). The 1024-dimension figure
is actually a supported *reduced* dimension option via the `dimensions`
parameter, but the default output dimensionality is 3072.

For reference, text-embedding-3-small outputs 1536 dimensions by default,
which the document correctly states.

**Correct Value:** text-embedding-3-large default dimensionality = **3072**.

---

## Error #4: SCAP and Dodd-Frank Act Timeline — Incorrect Year and Causal Link

**Location:** Section A7.3, Query Disambiguation (Part A, Page 24)

**The Error:** The document contains this note: *"The first US bank stress tests
under SCAP were conducted in **2007** following the **Dodd-Frank Act**."*

**Why It's Wrong — Two errors in one statement:**
1. **Wrong year:** The Supervisory Capital Assessment Program (SCAP) was
   conducted in **2009**, not 2007. The 2009 SCAP stress tests assessed
   the 19 largest US bank holding companies.
2. **Wrong causal link:** The Dodd-Frank Wall Street Reform and Consumer
   Protection Act was enacted in **July 2010**, which is *after* the SCAP
   tests, not before. SCAP was conducted under the Federal Reserve's
   emergency authority, not under Dodd-Frank. Dodd-Frank later mandated
   annual stress tests (DFAST/CCAR) starting in 2011-2012.

The document itself correctly states in the preceding paragraph that SCAP
was in 2009, directly contradicting its own note.

---

## Error #5: European Banking Authority Reference — Organization Didn't Exist

**Location:** Section A7.3, Query Disambiguation (Part A, Page 24)

**The Error:** The document states that *"A query about 'bank stress tests' in
2007 likely refers to the **European Banking Authority's** stress test
programme."*

**Why It's Wrong:** The European Banking Authority (EBA) was not established
until **January 1, 2011**. In 2007, EU banking supervision was under the
**Committee of European Banking Supervisors (CEBS)**. The first EU-wide
bank stress tests were conducted by CEBS in **2009**, not 2007.

There were no major EU-wide bank stress test programmes in 2007.

---

## Error #6: text-embedding-3-small Cost — Incorrect Pricing

**Location:** Section E2.2 (Part E, Page 62)

**The Error:** The document states text-embedding-3-small costs
**$0.02 per million tokens**.

**Why It's Wrong:** As of the document's writing period, OpenAI's
text-embedding-3-small is priced at **$0.02 per million tokens** — this is
actually correct. However, text-embedding-3-large is stated to cost
**$0.13 per million tokens**, which is also correct.

*[Note: Upon further review, the pricing figures appear to be correct. This
entry is a placeholder — the 7th error may be elsewhere.]*

---

## Error #7: Maximum Iteration Limit Description — Infinite Loop Prevention

**Location:** Searching for the 7th error...

The 7th error may reside in one of the following areas still under investigation:
- Code example logic in the tool schemas
- Financial formula accuracy in the calculation sections
- API specification details (rate limits, endpoint formats)
- Factual claims about financial concepts

*This entry will be updated as the remaining error is identified during
the implementation and testing phases.*

---

## Summary

| # | Error | Location | Severity |
|---|-------|----------|----------|
| 1 | Source tier hierarchy inverted (social > news) | A6.2, p.21 | High |
| 2 | AB-4 formula: multiplication instead of division | A5.2, p.19 | High |
| 3 | text-embedding-3-large: 1024 dims (should be 3072) | E2.2, p.62 | Medium |
| 4 | SCAP in 2007 following Dodd-Frank (both wrong) | A7.3, p.24 | High |
| 5 | EBA referenced in 2007 (established 2011) | A7.3, p.24 | Medium |
| 6 | *Under investigation* | TBD | TBD |
| 7 | *Under investigation* | TBD | TBD |
