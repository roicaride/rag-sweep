# rag-eval — pipeline de evaluación y optimización de RAG

Proyecto [Kedro](https://kedro.org) que orquesta toda la experimentación: desde la ingesta del
corpus bruto hasta las tablas comparativas finales. Es el núcleo reutilizable del trabajo; el
contexto y los resultados están en el [README raíz](../README.md).

## Requisitos

- Python ≥ 3.10
- Docker (para Qdrant en modo servidor)
- Claves de API para el LLM generador y el LLM juez

```bash
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

Las credenciales van en `conf/local/credentials.yml`, que **está excluido del control de
versiones**. Estructura esperada:

```yaml
deepseek:
  api_key: "..."      # LLM generador
google:
  api_key: "..."      # LLM juez de RAGAS (gemini-2.5-flash) y embedder juez
```

## Pipelines

| Pipeline | Qué hace | Entrada → salida |
|---|---|---|
| `ingestion` | Normaliza y limpia el volcado bruto de EUR-Lex | `raw_corpus` → `corpus_clean` |
| `corpus_selection` | Filtra el subcorpus de trabajo (palabra clave + tipo de acto + mínimo de palabras) | `corpus_clean` → `corpus_copyright` |
| `gold_dataset` | Muestreo estratificado + generación de pares QA con RAGAS | `corpus_copyright` → `gold_dataset` |
| `rag_base` | Chunking → indexado → recuperación → generación | `corpus_copyright` → `<exp>.rag_results` |
| `evaluation` | Calcula las 6 métricas RAGAS y las registra en MLflow | `<exp>.rag_results` → `<exp>.eval_metrics` |

Los pipelines `rag_base` y `evaluation` están parametrizados: se instancian una vez por cada
variante experimental declarada en `conf/base/parameters_exp*.yml`.

## Ejecución

Preparación del corpus y del conjunto de evaluación (una sola vez):

```bash
kedro run                              # ingestion + corpus_selection + gold_dataset
```

Barrida experimental, fase a fase. Tras cada fase se ejecuta el nodo de selección que fija el
ganador para las fases siguientes:

```bash
kedro run --pipeline exp0                          # baseline

kedro run --pipeline exp1__bge_m3                  # fase 1: embeddings
kedro run --pipeline exp1__qwen3_8b
kedro run --pipeline exp1__snowflake_arctic_l
kedro run --pipeline select_best_embedding

kedro run --pipeline exp2__fixed_512               # fase 2: chunking
# … resto de variantes exp2 …
kedro run --pipeline select_best_chunking

kedro run --pipeline exp3__mmr_k10                 # fase 3: retrieval
# … resto de variantes exp3 …
kedro run --pipeline select_best_retrieval

kedro run --pipeline exp4a__hyde                   # fase 4: expansión de consulta
# … resto de variantes exp4 …
kedro run --pipeline select_best_query_transform

kedro run --pipeline exp5b__rerank_ce_k10          # fase 5: reranking
# … resto de variantes exp5 …
kedro run --pipeline select_best_rerank

kedro run --pipeline expSOTA                       # control: stack de la literatura
```

Lista completa de pipelines disponibles: `kedro registry list`.
Grafo interactivo del flujo: `kedro viz`.
Métricas de todas las ejecuciones: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Añadir una variante experimental

Casi siempre basta con **un bloque YAML**. Para probar un embedding nuevo en la fase 1:

```yaml
# conf/base/parameters_exp1.yml
exp1__mi_modelo:
  experiment_id: "exp1__mi_modelo"
  chunking:  {strategy: fixed_size, chunk_size: 256, chunk_overlap: 25, tokenizer_model: "BAAI/bge-base-en-v1.5"}
  embedding: {model: "org/mi-modelo", batch_size: 32}
  retrieval: {strategy: cosine, top_k: 5}
  llm:       {model: "deepseek-v4-flash", temperature: 0.0, max_tokens: 2048, base_url: "https://api.deepseek.com"}
  ragas_llm: {model: "gemini-2.5-flash", temperature: 0.0, max_tokens: 4096, thinking_budget: 0}
```

Después hay que registrar la variante en `pipeline_registry.py` (una línea) y declarar sus tres
datasets en `conf/base/catalog.yml`.

Una clave omitida en el YAML **no** es un error: se resuelve en tiempo de ejecución al ganador
de la fase anterior. Por eso los bloques de `exp2` no fijan `embedding.model` — lo heredan de
`best_embedding`.

Para una **estrategia** que no exista (un tipo de chunking nuevo, un retriever distinto), el
punto de extensión es el módulo correspondiente en `src/rag_eval/`.

## Estructura del código

```
src/rag_eval/
├── pipeline_registry.py   Registro de pipelines y nodos select_best_* (arrastre de ganadores)
├── selection.py           Lógica de elección del ganador de cada fase por P+R
├── chunking.py            fixed_size · sentence · structural · parent-child · semantic
├── embedding.py           Carga de embedders (HF Inference API, Scaleway, local)
├── retrieval.py           coseno · MMR · BM25 · híbrido (RRF) · reranking CE y LLM
├── query_transform.py     HyDE · Multi-Query · RAG-Fusion · Rewrite · Step-Back · Decomposition · Self-Query
├── datasets.py            Datasets Kedro a medida (JSONL, knowledge graph de RAGAS)
├── hooks.py               Integración con MLflow
└── pipelines/
    ├── ingestion/ · corpus_selection/ · gold_dataset/ · rag_base/ · evaluation/
```

## Datos

`data/` sigue la convención de capas de Kedro (`01_raw` → `08_reporting`) y **está excluido del
repositorio**: son ~11 GB entre el corpus y los índices vectoriales, todos regenerables.

Sí están versionadas las figuras y tablas finales, copiadas a
[`resultados/`](../resultados/) en la raíz del proyecto.

## Notebooks

| Notebook | Contenido |
|---|---|
| `notebooks/00_corpus_exploration.ipynb` | Análisis exploratorio del corpus EUR-Lex completo |
| `notebooks/01_copyright_corpus.ipynb` | Caracterización del subcorpus de derecho de autor |
| `notebooks/02_gold_dataset.ipynb` | Validación del conjunto de evaluación generado |
| `notebooks/03_results.ipynb` | Genera todas las figuras y tablas comparativas |

`info/` contiene los notebooks precursores: el prototipo manual del flujo RAG, previo a su
formalización como pipeline Kedro. Se conservan como registro del desarrollo.
