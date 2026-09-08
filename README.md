# rag-sweep

**A reproducible framework for finding the RAG configuration that actually works on *your* corpus — and its application to a European legal domain.**

Bachelor's thesis · Business & Technology · Universidade de Santiago de Compostela
Roi Caride Borrajo · Thesis written in Galician · 2026

---

## What this is

There is no universally optimal RAG configuration. What works on a corpus of technical
documentation has no reason to work on a legal, medical or financial one, and the only honest
way to find out is **to measure it on your own corpus**.

This repository contains **the tool that makes that measurement systematic and repeatable**: a
sequential ablation pipeline built on Kedro that compares embedding, chunking, retrieval, query
expansion and reranking variants, evaluates each one against the same metrics on the same set
of questions, and **automatically carries the winner of each stage forward as the starting
point of the next**.

> **The results you will find here are a case study, not the product.**
> The sweep ran on a corpus of EU copyright law because it had to run on something. The
> specific findings — that `bge-m3` wins, that MMR with k=10 wins, that no query expansion
> technique helps — **belong to that corpus and should not be extrapolated**. What transfers is
> the procedure: swap the corpus and the parameter files, run it again, and you get the optimal
> configuration *for your domain* with the same evidence behind it.

---

## Why the tool is needed

The most telling experiment in this work is not which configuration won, but this comparison:

| Configuration | P+R |
|---|---|
| Naive RAG (baseline) | 0.756 |
| **"SOTA" stack assembled from the literature** | **1.095** |
| **Configuration found empirically on this corpus** | **1.203** |

A stack built by piling up the techniques that papers report as state of the art (semantic
chunking + an 8B-parameter embedding model + hybrid retrieval + RAG-Fusion + LLM reranking)
**performs 9% worse** than the configuration found by measuring on the actual corpus — and it
is considerably more expensive to run.

That gap is the argument of this project: following generic recommendations from the
literature is not the same as optimizing. You need a method, and the method needs
instrumentation.

---

## How it works

**Sequential ablation with winner carry-over.** Each stage varies a single component and holds
everything else fixed. When it finishes, a selection node picks the winner by the decision
metric and injects it as the base configuration for the next stage.

```
exp0  baseline          →  fixed 256 · bge-base-en-v1.5 · cosine k=5
  ↓  select_best_embedding
exp1  embeddings        →  bge-m3 vs qwen3-8b vs snowflake-arctic-l
  ↓  select_best_chunking
exp2  chunking          →  fixed 512 · sentence · structural · structural+parent-child · semantic
  ↓  select_best_retrieval
exp3  retrieval         →  dense k3/k10 · MMR k5/k10 · BM25 k5/k10 · hybrid k5/k10
  ↓  select_best_query_transform
exp4  query expansion   →  HyDE · Multi-Query · RAG-Fusion · Rewrite · Step-Back · Decomposition · Self-Query
  ↓  select_best_rerank
exp5  reranking         →  cross-encoder k5/k10 · LLM-as-reranker k5/k10

expSOTA  control: maximalist stack from the literature, run separately for contrast
```

The carry-over lives in [`pipeline_registry.py`](rag-eval/src/rag_eval/pipeline_registry.py):
the `select_best_*` nodes read the metrics of every variant in a stage and produce a dataset
(`best_embedding`, `best_chunking`, …) that later stages receive as input. Adding a new variant
means **adding a block to a YAML file**, not touching code.

Stages 3, 4 and 5 reuse the vector index built in stage 2 instead of re-indexing, which cuts
the cost of a full sweep from hours to minutes.

**A known and accepted limitation:** greedy stage-by-stage optimization does not guarantee a
global optimum — A₁ winning the first stage does not imply that A₁+B₁ beats A₂+B₂. This is a
sequential hyperparameter search, chosen because an exhaustive search over the full
combinatorial space was not feasible within the budget of this work.

---

## Metrics

Six RAGAS metrics, with a fixed judge LLM (`gemini-2.5-flash`) kept independent from the
generator LLM (`deepseek-v4-flash`) to avoid self-evaluation bias:

| Metric | What it measures | Stage |
|---|---|---|
| `context_recall` | Was everything needed to answer actually retrieved? | Retrieval |
| `context_precision` | Is what was retrieved relevant, and well ranked? | Retrieval |
| `context_entity_recall` | Coverage of the key entities of the ideal answer | Retrieval |
| `faithfulness` | Is the answer grounded in the retrieved context? | Generation |
| `answer_relevancy` | Does the answer address the question asked? | Generation |
| `noise_sensitivity` | Does it introduce errors when given irrelevant context? | Robustness |

**Decision metric: `P+R` = `context_precision` + `context_recall`.** Retrieval is what the
pipeline directly controls, so that is what gets optimized; generation quality is monitored as
a side effect, not as the objective.

---

## Repository layout

```
.
├── rag-eval/            Kedro pipeline — the reusable core
│   ├── conf/base/       parameters_exp*.yml → one sweep = one config file
│   ├── src/rag_eval/    5 pipelines + chunking, embedding, retrieval, reranking modules
│   ├── notebooks/       corpus exploration, gold set construction, final analysis
│   └── info/            precursor notebooks (prototype that predates the pipeline)
├── results/             EUR-Lex case study: 22 figures + 10 CSV tables
├── thesis/              LaTeX source, bibliography, figures and final PDF
└── docs/
    ├── experimental-design.md   The executed design, stage by stage
    ├── evaluation-dataset.md    How the evaluation set was built
    └── specs/                   Technical decisions recorded during development
```

Not in the repository, because it is heavy and regenerable: the raw corpus (3.1 GB), the
cleaned corpus (2.1 GB), the Qdrant indexes (5.8 GB) and the MLflow store. The pipeline
rebuilds all of it from the public source.

---

## The case study: EUR-Lex

| | |
|---|---|
| Source corpus | `gplsi/alia_intellectual_property` (EUR-Lex, intellectual property, CC BY 4.0) — 40,181 documents |
| Working subcorpus | 363 copyright documents, filtered by title keyword, legal act type and a 300-word minimum |
| Evaluation set | 50 question–answer pairs generated with RAGAS (26 single-hop, 24 multi-hop), stratified sampling by length, `seed=42` |
| Configurations evaluated | 29 full runs across 6 stages |
| Vector store | Qdrant in server mode |
| Orchestration / tracking | Kedro 1.3.1 · MLflow |

### Pipeline progression

| Stage | Decision | P+R | Δ |
|---|---|---|---|
| Baseline | fixed 256 · bge-base-en-v1.5 · cosine k=5 | 0.756 | — |
| + Embedding | **bge-m3** | 0.999 | +0.242 |
| + Chunking | fixed 256 *(the baseline wins)* | 0.999 | 0.000 |
| + Retrieval | **MMR k=10** | 1.094 | +0.095 |
| + Query expansion | **none** *(all 7 techniques hurt)* | 1.094 | 0.000 |
| + Reranking | **cross-encoder `bge-reranker-v2-m3` k=10** | **1.203** | +0.109 |

**+59% over the baseline.** Two of the five stages contributed nothing: the baseline chunking
turned out to be the best available, and **all seven query expansion techniques made results
worse without exception** — a negative finding most likely explained by how specific legal
vocabulary is, where rewriting the query moves it away from the corpus's literal terminology.

Every figure comes from [`results/tables/`](results/tables/) and matches what the thesis
reports.

---

## Using it on another domain

1. Replace the corpus in `rag-eval/data/01_raw/` and adjust the `corpus_selection` block of
   `conf/base/parameters.yml` with the filters for your domain.
2. Generate the evaluation set: `kedro run --pipeline gold_dataset`.
3. Run the baseline and whichever stages interest you. Variants are declared in
   `conf/base/parameters_exp*.yml`; testing a new embedding model is a matter of copying a
   block and changing the model name.
4. `notebooks/03_results.ipynb` regenerates every comparative figure and table.

No Python changes are needed unless you want to add a **strategy** that does not exist yet (a
new chunking method, for instance); in that case the extension point is the corresponding
module in `src/rag_eval/`.

Full installation and execution instructions: [`rag-eval/README.md`](rag-eval/README.md).

---

## Scope and honesty

This is a **reusable experimental asset**, not a deployed product. Specifically:

- **There is no service, API or user interface.** The pipeline produces evidence, not a
  production system.
- **Latency and production cost were not measured.** They were left out of scope.
- **Alternative architectures were not evaluated** (CRAG, Adaptive-RAG, agentic RAG,
  GraphRAG). They were in the original plan and remain future work.
- **The results hold for this corpus.** That is precisely the point of the work.

---

## Thesis

[`thesis/TFG_RAG_Roi_Caride.pdf`](thesis/TFG_RAG_Roi_Caride.pdf) — 48 pages, written in
Galician. LaTeX source included (`TFG_RAG.tex` + `referencias.bib`); build with
`pdflatex → biber → pdflatex → pdflatex`.

## Credits

Corpus derived from [`gplsi/alia_intellectual_property`](https://huggingface.co/datasets/gplsi/alia_intellectual_property)
(Espinosa Zaragoza et al., 2025), CC BY 4.0. Built with [Kedro](https://kedro.org),
[LangChain](https://www.langchain.com), [RAGAS](https://docs.ragas.io),
[Qdrant](https://qdrant.tech) and [MLflow](https://mlflow.org).
