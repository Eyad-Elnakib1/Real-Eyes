# ADR-002: Wikipedia WP:RSP for Source Credibility Scoring

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07 |
| **Decision** | Use Wikipedia's Perennial Sources (WP:RSP) list as the backbone for source trust scoring |
| **Deciders** | RealEyes Engineering Team |

---

## Context

The fact-checking pipeline retrieves evidence from the open web via DuckDuckGo. Not all sources are equally reliable — a claim "supported" by InfoWars should weigh less than the same claim supported by Reuters. We need a trust-scoring mechanism that:

1. Covers a wide range of domains (news, academic, social, fringe).
2. Has clear, defensible tier classifications.
3. Is maintained by a neutral community (not our own subjective judgment).
4. Can be used for both **search prioritization** (fetch trusted sources first) and **verdict weighting** (trusted agreement counts more).

## Decision

We adopted the **Wikipedia Reliable Sources/Perennial Sources (WP:RSP)** list as implemented in `semantic_crawler/utils/trust.py`:

- **`RSP_TRUSTED`** (~166 domains): Sources classified as "generally reliable" by Wikipedia editors. Used for site:-restricted DuckDuckGo queries (trusted-first search).
- **`RSP_BLOCKED`** (~80 domains): Sources classified as "deprecated" or "generally unreliable." Silently dropped from search results.
- **`trust_score(url)`**: Returns a float multiplier used in NLI aggregation:
  - `3.0` for fact-checking organizations (Snopes, PolitiFact, Reuters Fact Check)
  - `2.0` for WP:RSP generally-reliable sources
  - `1.0` for neutral/unknown domains
  - `0.2` for social media platforms (Reddit, Twitter, Facebook)
  - `0.0` for blocked domains (InfoWars, Natural News, state propaganda)

## Rationale

### 1. Academic Defensibility

WP:RSP is arguably the most thoroughly debated and documented source reliability classification in existence. Each source's status is decided by community consensus with extensive talk-page discussions. This makes our trust scoring **reproducible and citable** in academic publications.

### 2. Coverage

The list covers 166+ generally-reliable domains spanning:
- Major wire services (AP, Reuters, AFP)
- National newspapers (NYT, WSJ, Guardian, BBC)
- Fact-checkers (Snopes, PolitiFact, FactCheck.org)
- Academic/scientific publishers
- Government agencies (WHO, CDC, NASA)

### 3. Dual Use in the Pipeline

The same trust scores serve two roles:
1. **Search prioritization:** Trusted domains are searched first via site:-restricted queries. This ensures they occupy the front slots of the evidence pool even if DuckDuckGo's open-web ranking would have buried them.
2. **Verdict weighting:** In the NLI aggregation formula, each evidence chunk's NLI probability is multiplied by `retrieval_score × credibility`. A chunk from Reuters (credibility=2.0) contributes 2× the weight of an unknown blog.

### 4. Community Maintenance

We don't have to maintain the list ourselves. As Wikipedia's community updates WP:RSP (which happens continuously), we can periodically refresh our local copy.

## Consequences

### Positive
- Objective, third-party source classification — no accusations of bias.
- High coverage of English-language sources.
- Single source of truth shared across crawler, retriever, and checker.

### Negative
- WP:RSP is English-centric. Arabic sources require manual additions to `_TRUSTED_SEARCH_DOMAINS`.
- Some legitimate sources may be classified as "no consensus" and receive neutral scoring.
- The list can change — a refresh strategy is needed.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Media Bias/Fact Check (MBFC)** | Proprietary ratings, controversial methodology, no API |
| **NewsGuard** | Paid service, not freely redistributable |
| **Custom hand-curated list** | Subjective, hard to defend academically, maintenance burden |
| **No trust scoring** | InfoWars and Reuters would count equally — unacceptable |
