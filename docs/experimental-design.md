# Experimental design

A description of the sweep **as it was actually executed**. Deviations from the original plan
are collected at the end.

## Principle: sequential ablation with carry-over

Each stage varies **a single component** and holds everything else fixed (*ceteris paribus*).
When it finishes, a `select_best_*` node reads the metrics of every variant, picks the winner
by `P+R`, and injects it as the base configuration for the next stage.

This is visible in the parameter files themselves: `exp2` blocks **do not declare**
`embedding.model`. The missing key resolves at runtime to the value of `best_embedding`. Each
stage inherits the decisions already made without anything being rewritten by hand.

**Decision metric:** `P+R` = `context_precision` + `context_recall`. Retrieval is what gets
optimized because it is what the pipeline directly controls; generation metrics
(`faithfulness`, `answer_relevancy`) are monitored as a side effect.

**Accepted limitation:** this is greedy optimization. A₁ winning the first stage does not
guarantee that A₁+B₁ beats A₂+B₂; a stepwise optimum need not be the global one. It was chosen
because an exhaustive search over the full combinatorial space was not feasible within the
compute and API budget of this work.

---

## Stage 0 — Baseline

The reference point everything else is measured against. Deliberately plain:

| Component | Value |
|---|---|
| Chunking | `fixed_size`, 256 tokens, 25 overlap (~10%) |
| Embedding | `BAAI/bge-base-en-v1.5` (86M parameters, 768 dim) |
| Retrieval | Cosine, `top_k=5` |
| Generator LLM | `deepseek-v4-flash`, temperature 0 |
| Judge LLM | `gemini-2.5-flash`, temperature 0 |

**Result: P+R = 0.756.**

The generator and the judge are pinned here and **never change in any later experiment**: they
are control variables. Without an honest baseline there is no way to tell whether anything
improved or regressed.

---

## Stage 1 — Embeddings

Only the embedding model varies. Chunking and retrieval stay nailed to the baseline.

| Variant | Model | P+R |
|---|---|---|
| *(= exp0)* | `BAAI/bge-base-en-v1.5` | 0.756 |
| `exp1__bge_m3` | **`BAAI/bge-m3`** | **0.999** |
| `exp1__qwen3_8b` | `Qwen/Qwen3-Embedding-8B` | 0.889 |
| `exp1__snowflake_arctic_l` | `Snowflake/snowflake-arctic-embed-l-v2.0` | 0.775 |

**Winner: `bge-m3` (+0.242).** The single largest jump of the entire sweep.

A notable result: `Qwen3-Embedding-8B`, far higher on the MTEB leaderboard and nearly a hundred
times larger, lands behind `bge-m3` on this corpus. The first sign that positions on
general-purpose leaderboards do not transfer cleanly to a specialized domain.

`bge-base` is not repeated as a variant: it is identical to exp0 and that column is reused.

---

## Stage 2 — Chunking

Embedding pinned to the stage 1 winner (`bge-m3`, resolved automatically).

| Variant | Strategy | P+R |
|---|---|---|
| *(= exp1\_\_bge_m3)* | **`fixed_size` 256** | **0.999** |
| `exp2__sentence` | `sentence` 512 | 0.993 |
| `exp2__structural_pc` | `structural` + parent-child | 0.958 |
| `exp2__fixed_512` | `fixed_size` 512 | 0.913 |
| `exp2__semantic` | `semantic` (percentile) | 0.891 |
| `exp2__structural` | `structural` | 0.729 |

**Winner: the baseline chunking (fixed 256). Δ = 0.000.**

None of the five alternatives beats the simplest possible fixed-size split. Semantic chunking,
the most sophisticated of the set, comes fifth out of six. Short chunks (256 rather than 512)
work better because they concentrate signal: in a legal corpus, a long chunk mixes several
provisions together and dilutes similarity with the query.

---

## Stage 3 — Retrieval

Embedding and chunking already fixed. This stage **reuses the stage 2 index** instead of
re-indexing, which makes the sweep dramatically cheaper.

| Variant | Strategy | `top_k` | P+R |
|---|---|---|---|
| `exp3__mmr_k10` | **MMR (`fetch_k=30`)** | **10** | **1.094** |
| `exp3__dense_k10` | Cosine | 10 | 1.050 |
| `exp3__mmr` | MMR | 5 | 1.023 |
| *(= exp1\_\_bge_m3)* | Cosine | 5 | 0.999 |
| `exp3__dense_k3` | Cosine | 3 | 0.915 |
| `exp3__bm25_k10` | BM25 | 10 | 0.914 |
| `exp3__hybrid_bm25` | Hybrid dense + BM25 (RRF) | 5 | 0.898 |
| `exp3__hybrid_k10` | Hybrid dense + BM25 (RRF) | 10 | 0.886 |
| `exp3__bm25` | BM25 | 5 | 0.808 |

**Winner: MMR with `top_k=10` (+0.095).**

Two readings. First, widening `top_k` pays off. In the three strategies where retrieval is
purely ranking-based — MMR, cosine and BM25 — going from 5 to 10 improves P+R: recall gains
more than precision loses. The one exception is the hybrid retriever, where the opposite
happens (0.898 at k=5 against 0.886 at k=10).

Second, hybrid retrieval — an almost universal recommendation in the literature — is among the
worst performers here. BM25 adds little on a corpus with heavily normalized terminology, and
fusing it via RRF degrades the dense ranking instead of complementing it.

---

## Stage 4 — Query expansion

Seven techniques, all inheriting the winning embedding, chunking and retrieval.

| Variant | Technique | P+R |
|---|---|---|
| *(= exp3\_\_mmr_k10)* | **None** | **1.094** |
| `exp4b__multiquery` | Multi-Query | 1.047 |
| `exp4c__rag_fusion` | RAG-Fusion | 1.015 |
| `exp4e__stepback` | Step-Back | 1.003 |
| `exp4d__rewrite` | Rewrite | 0.987 |
| `exp4f__decomposition` | Decomposition | 0.942 |
| `exp4a__hyde` | HyDE | 0.846 |
| `exp4g__selfquery` | Self-Query | 0.823 |

**Winner: none. Δ = 0.000.**

All seven make results worse, without exception. It is the clearest negative finding of the
work.

The likely explanation is vocabulary: the legal corpus uses literal, heavily normalized
terminology ("related right", "computer program"), and every expansion technique reformulates
the query in the LLM's own language, moving it away from that literalness. HyDE, which
generates a full hypothetical document, does the most damage — precisely the most aggressive
reformulation of the set.

---

## Stage 5 — Reranking

| Variant | Reranker | Final `top_k` | P+R |
|---|---|---|---|
| `exp5b__rerank_ce_k10` | **Cross-encoder `BAAI/bge-reranker-v2-m3`** | **10** | **1.203** |
| `exp5a__rerank_ce_k5` | Cross-encoder `bge-reranker-v2-m3` | 5 | 1.187 |
| `exp5c__rerank_llm_k10` | LLM listwise (`deepseek-v4-flash`) | 10 | 1.184 |
| `exp5d__rerank_llm_k5` | LLM listwise (`deepseek-v4-flash`) | 5 | 1.154 |
| *(= exp3\_\_mmr_k10)* | No reranking | 10 | 1.094 |

**Winner: cross-encoder with `top_k=10` (+0.109).**

Reranking runs as a cascade: a pool of 50 candidates → 20 selected by MMR → reranker → final
top-k. The cross-encoder, far cheaper than the LLM reranker, beats it in all four comparisons.

---

## Control: the "SOTA" stack

Run separately, without carry-over: it combines the **most advanced technique on paper** for
each dimension rather than the empirical winner.

| Dimension | SOTA stack | Empirical configuration |
|---|---|---|
| Embedding | `Qwen3-Embedding-8B` (top MTEB) | `bge-m3` |
| Chunking | `semantic` | `fixed_size` 256 |
| Retrieval | Hybrid dense + BM25 (RRF) | MMR `k=10` |
| Query expansion | RAG-Fusion | None |
| Reranking | LLM listwise (RankGPT) | Cross-encoder `bge-reranker-v2-m3` |

| Configuration | P+R |
|---|---|
| Naive RAG (exp0) | 0.756 |
| SOTA stack (expSOTA) | 1.095 |
| **Empirical configuration (exp5b)** | **1.203** |

The SOTA stack does improve on the baseline, but it **falls 9% short** of the configuration
found by measuring — and it is appreciably more expensive to run (an 8B embedding model,
LLM-driven query expansion and generative reranking).

This contrast is what justifies the whole framework: stacking recommendations from the
literature is not the same as optimizing. You have to measure on your own corpus.

---

## Deviations from the original plan

| Original plan | Executed | Reason |
|---|---|---|
| 7 RAGAS metrics + latency | **6 metrics**: `context_recall`, `context_precision`, `context_entity_recall`, `faithfulness`, `answer_relevancy`, `noise_sensitivity` | `factual_correctness` and `semantic_similarity` were dropped; latency fell out of scope |
| Baseline with `bge-small`, fixed 512 | `bge-base-en-v1.5`, fixed 256 | A more reasonable baseline makes the comparison harder to win, and therefore more honest |
| Multilingual embeddings (`multilingual-e5`, `paraphrase-multilingual`) | `bge-m3`, `qwen3-8b`, `snowflake-arctic-l` | The corpus is monolingual English; leading English models were prioritized instead |
| Reranking inside stage 4 | Its own stage 5, separate from query expansion | They are different pipeline stages (pre- and post-retrieval) and mixing them broke *ceteris paribus* |
| Stage 4b: CRAG, Adaptive-RAG, agentic RAG, GraphRAG | **Not executed** | Outside the time budget; left as future work |
| — | **expSOTA** (control against the literature) | Added during execution; ended up being the most relevant result of the work |

The original plan is described in the thesis. These deviations are documented because the
route taken — including the stages that contributed nothing — is part of the result.
