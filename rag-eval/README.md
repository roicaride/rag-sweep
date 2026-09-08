# rag-eval — RAG evaluation and optimization pipeline

A [Kedro](https://kedro.org) project that orchestrates the whole experiment: from ingesting the
raw corpus to the final comparison tables. This is the reusable core of the work; context and
results live in the [root README](../README.md).

## Requirements

- Python ≥ 3.10
- Docker (for Qdrant in server mode)
- API keys for the generator LLM and the judge LLM

```bash
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

Credentials go in `conf/local/credentials.yml`, which is **excluded from version control**.
Expected structure:

```yaml
deepseek:
  api_key: "..."      # generator LLM
google:
  api_key: "..."      # RAGAS judge LLM (gemini-2.5-flash) and judge embedder
```

## Pipelines

| Pipeline | What it does | Input → output |
|---|---|---|
| `ingestion` | Normalizes and cleans the raw EUR-Lex dump | `raw_corpus` → `corpus_clean` |
| `corpus_selection` | Filters the working subcorpus (keyword + legal act type + minimum length) | `corpus_clean` → `corpus_copyright` |
| `gold_dataset` | Stratified sampling + QA pair generation with RAGAS | `corpus_copyright` → `gold_dataset` |
| `rag_base` | Chunking → indexing → retrieval → generation | `corpus_copyright` → `<exp>.rag_results` |
| `evaluation` | Computes the 6 RAGAS metrics and logs them to MLflow | `<exp>.rag_results` → `<exp>.eval_metrics` |

`rag_base` and `evaluation` are parameterized: they are instantiated once per experimental
variant declared in `conf/base/parameters_exp*.yml`.

## Running it

Corpus and evaluation set preparation (once):

```bash
kedro run                              # ingestion + corpus_selection + gold_dataset
```

The experimental sweep, stage by stage. After each stage, the selection node fixes the winner
for everything downstream:

```bash
kedro run --pipeline exp0                          # baseline

kedro run --pipeline exp1__bge_m3                  # stage 1: embeddings
kedro run --pipeline exp1__qwen3_8b
kedro run --pipeline exp1__snowflake_arctic_l
kedro run --pipeline select_best_embedding

kedro run --pipeline exp2__fixed_512               # stage 2: chunking
# … remaining exp2 variants …
kedro run --pipeline select_best_chunking

kedro run --pipeline exp3__mmr_k10                 # stage 3: retrieval
# … remaining exp3 variants …
kedro run --pipeline select_best_retrieval

kedro run --pipeline exp4a__hyde                   # stage 4: query expansion
# … remaining exp4 variants …
kedro run --pipeline select_best_query_transform

kedro run --pipeline exp5b__rerank_ce_k10          # stage 5: reranking
# … remaining exp5 variants …
kedro run --pipeline select_best_rerank

kedro run --pipeline expSOTA                       # control: stack from the literature
```

Full list of available pipelines: `kedro registry list`.
Interactive flow graph: `kedro viz`.
Metrics across all runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Adding an experimental variant

Usually **one YAML block** is enough. To test a new embedding model in stage 1:

```yaml
# conf/base/parameters_exp1.yml
exp1__my_model:
  experiment_id: "exp1__my_model"
  chunking:  {strategy: fixed_size, chunk_size: 256, chunk_overlap: 25, tokenizer_model: "BAAI/bge-base-en-v1.5"}
  embedding: {model: "org/my-model", batch_size: 32}
  retrieval: {strategy: cosine, top_k: 5}
  llm:       {model: "deepseek-v4-flash", temperature: 0.0, max_tokens: 2048, base_url: "https://api.deepseek.com"}
  ragas_llm: {model: "gemini-2.5-flash", temperature: 0.0, max_tokens: 4096, thinking_budget: 0}
```

Then register the variant in `pipeline_registry.py` (one line) and declare its three datasets
in `conf/base/catalog.yml`.

A key omitted from the YAML is **not** an error: it resolves at runtime to the winner of the
previous stage. That is why `exp2` blocks do not pin `embedding.model` — they inherit it from
`best_embedding`.

For a **strategy** that does not exist yet (a new chunking method, a different retriever), the
extension point is the corresponding module in `src/rag_eval/`.

## Code layout

```
src/rag_eval/
├── pipeline_registry.py   Pipeline registry and select_best_* nodes (winner carry-over)
├── selection.py           Logic that picks each stage's winner by P+R
├── chunking.py            fixed_size · sentence · structural · parent-child · semantic
├── embedding.py           Embedder loading (HF Inference API, Scaleway, local)
├── retrieval.py           cosine · MMR · BM25 · hybrid (RRF) · CE and LLM reranking
├── query_transform.py     HyDE · Multi-Query · RAG-Fusion · Rewrite · Step-Back · Decomposition · Self-Query
├── datasets.py            Custom Kedro datasets (JSONL, RAGAS knowledge graph)
├── hooks.py               MLflow integration
└── pipelines/
    ├── ingestion/ · corpus_selection/ · gold_dataset/ · rag_base/ · evaluation/
```

## Data

`data/` follows the Kedro layer convention (`01_raw` → `08_reporting`) and is **excluded from
the repository**: roughly 11 GB of corpus and vector indexes, all regenerable.

The final figures and tables are versioned, copied to [`results/`](../results/) at the project
root.

## Notebooks

| Notebook | Contents |
|---|---|
| `notebooks/00_corpus_exploration.ipynb` | Exploratory analysis of the full EUR-Lex corpus |
| `notebooks/01_copyright_corpus.ipynb` | Characterization of the copyright subcorpus |
| `notebooks/02_gold_dataset.ipynb` | Validation of the generated evaluation set |
| `notebooks/03_results.ipynb` | Generates every comparative figure and table |

`info/` holds the precursor notebooks: the manual prototype of the RAG flow, predating its
formalization as a Kedro pipeline. Kept as a record of how the project developed.
