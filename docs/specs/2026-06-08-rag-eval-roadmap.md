# RAG Eval — Diseño y Roadmap
*TFG · 2026-06-08*

---

## 1. Objetivo

Evaluar sistemáticamente distintas configuraciones de RAG sobre el corpus EUR-Lex de derecho de autor europeo, usando un gold dataset de 149 pares QA generados con RAGAS. Los resultados orientan qué combinación de chunking + embedding + retrieval maximiza la calidad de respuesta para preguntas legales.

---

## 2. Estado actual

| Artefacto | Estado |
|---|---|
| `data/01_raw/eurlex-en-md.jsonl` | Corpus crudo EUR-Lex |
| `data/02_intermediate/corpus_clean.jsonl` | Texto limpiado (markdown, whitespace) |
| `data/02_intermediate/corpus_copyright.jsonl` | 363 docs filtrados por copyright |
| `data/03_primary/gold_dataset.json` | 149 QA pairs (75 single-hop + 74 multi-hop) |
| `pipelines/ingestion` | Implementado |
| `pipelines/corpus_selection` | Implementado |
| `pipelines/gold_dataset` | Implementado |
| `pipelines/rag_base` | **Vacío — por implementar** |
| `pipelines/evaluation` | **Vacío — por implementar** |

---

## 3. Métricas de evaluación (6 — todas RAGAS nativas)

| Métrica | Mide | Inputs |
|---|---|---|
| `ContextPrecision` | Proporción del contexto recuperado que es relevante | user_input + retrieved_contexts + reference |
| `ContextRecall` | Grado en que se recupera la información necesaria | user_input + retrieved_contexts + reference |
| `ContextEntityRecall` | Entidades clave recuperadas | retrieved_contexts + reference |
| `Faithfulness` | Respuesta basada en el contexto (sin alucinaciones) | user_input + response + retrieved_contexts |
| `ResponseRelevancy` | Respuesta responde a la pregunta | user_input + response |
| `NoiseSensitivity` | Impacto del ruido en el contexto sobre la respuesta | user_input + response + reference + retrieved_contexts |

**Importante:** `retrieved_contexts` viene siempre del pipeline RAG en tiempo de evaluación, nunca del gold dataset. El gold dataset aporta `user_input` y `reference` (respuesta ground truth).

---

## 4. Stack técnico

| Componente | Elección | Justificación |
|---|---|---|
| Framework de pipelines | Kedro 1.3.1 | Ya instalado |
| Vector store | FAISS (local) | Offline, rápido, soporta MMR, fácil save/load |
| LLM generador | DeepSeek V4 Flash | Ya integrado en gold pipeline, API OpenAI-compatible |
| LLM juez (RAGAS) | DeepSeek V4 Flash | Mismo modelo, consistencia |
| Embedding baseline | Gemini Embedding 2 | Ya integrado |
| Experiment tracking | MLflow directo en nodos | kedro-mlflow no soporta Kedro 1.3.x |
| Chunking | LangChain splitters + custom | Ver Fase 1 |
| Reranking | sentence-transformers (bge-reranker-v2-m3) | SOTA multilingüe |

---

## 5. Flujo de datos end-to-end

```
corpus_copyright.jsonl (363 docs)
    │
    ▼ [chunk_documents]   ← strategy, chunk_size, overlap
    │  chunks: List[Document]
    │
    ▼ [build_index]        ← embedding_model
    │  FAISSDataset → data/04_feature/{exp_id}/faiss_index/
    │
    ▼ [run_rag]            ← gold_dataset (149 queries), retrieval params, LLM
    │  rag_results.json → data/07_model_output/{exp_id}/
    │  campos: user_input, retrieved_contexts, response, reference
    │
    ▼ [evaluate_ragas]     ← 6 métricas + mlflow.log_params/metrics()
       metrics.parquet → data/08_reporting/{exp_id}/
       MLflow run: parámetros + 6 métricas
```

**Separación build_index / run_rag:** permite reutilizar el mismo índice para probar distintas estrategias de retrieval sin re-indexar.

---

## 6. Nodos core (reutilizados en todos los experimentos)

### `chunk_documents(corpus, params) → List[Document]`
Aplica la estrategia de chunking configurada. Devuelve LangChain `Document` objects.

### `build_index(chunks, params) → FAISS`
Instancia el embedding model, construye el índice FAISS. Se persiste en disco con `save_local()`.

### `run_rag(vectorstore, gold_dataset, params) → List[dict]`
Para cada query del gold dataset:
1. Recupera chunks con la estrategia de retrieval configurada
2. Genera respuesta con DeepSeek vía LangChain RAG chain
3. Guarda `{user_input, retrieved_contexts, response, reference}`

### `evaluate_ragas(rag_results, params) → pd.DataFrame`
Construye `EvaluationDataset`, llama a `ragas.evaluate()` con las 6 métricas, llama a `mlflow.log_params()` + `mlflow.log_metrics()`, devuelve DataFrame con scores.

---

## 7. Datasets Kedro custom necesarios

### `FAISSDataset`
```python
# src/rag_eval/datasets.py (añadir)
class FAISSDataset(AbstractDataset):
    def __init__(self, filepath, embedding_model_name, **kwargs):
        self._filepath = Path(filepath)
        self._embedding_model_name = embedding_model_name

    def _load(self) -> FAISS:
        embeddings = _build_embeddings(self._embedding_model_name)
        return FAISS.load_local(str(self._filepath), embeddings,
                                allow_dangerous_deserialization=True)

    def _save(self, vs: FAISS) -> None:
        self._filepath.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self._filepath))
```

El `embedding_model_name` se pasa desde `catalog.yml` como string y se instancia internamente.

---

## 8. Estructura Kedro — namespaces

Cada variante de experimento = un namespace. El `pipeline_registry.py` genera pipelines a partir de los ficheros `parameters_exp*.yml` existentes.

```python
# pipeline_registry.py
def make_rag_pipeline(namespace: str) -> Pipeline:
    return pipeline(
        [chunk_node, build_index_node, run_rag_node, evaluate_node],
        namespace=namespace,
        parameters={f"params:{k}": k for k in ["chunking", "embedding", "retrieval", "llm"]},
    )

pipelines["exp0"]                  = make_rag_pipeline("exp0")
pipelines["exp1__bge_base_en_v15"] = make_rag_pipeline("exp1__bge_base_en_v15")
# etc.
```

Ejecución: `kedro run --pipeline exp1__bge_base_en_v15` → un run de MLflow.

---

## 9. Fases de experimentación

### FASE 0 — Baseline (obligatorio)

| Param | Valor |
|---|---|
| chunking | fixed_size, chunk_size=512, overlap=50 |
| embedding | gemini-embedding-2 |
| retrieval | cosine dense, top_k=5 |
| llm | deepseek-v4-flash |

Establece el punto de referencia. Todos los experimentos siguientes se comparan contra este.

---

### FASE 1 — Chunking (impacto alto · bajo coste)

Embedding fijado al baseline. Retrieval fijado al baseline.

| Experimento | Estrategia | Implementación |
|---|---|---|
| `exp1__fixed_256` | Fixed 256 tokens, overlap 25 | `RecursiveCharacterTextSplitter` |
| `exp1__fixed_512` | Fixed 512 tokens, overlap 50 (= baseline) | `RecursiveCharacterTextSplitter` |
| `exp1__fixed_1024` | Fixed 1024 tokens, overlap 100 | `RecursiveCharacterTextSplitter` |
| `exp1__recursive` | Recursive con separadores semánticos | `RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," "])` |
| `exp1__structural` | Splits por estructura EUR-Lex (artículos, párrafos numerados, secciones) | Custom `EURLexStructuralSplitter` |
| `exp1__semantic` | Cortes donde cambia el tema (embedding-based) | `SemanticChunker` (langchain_experimental) |

**`EURLexStructuralSplitter`** — nodo custom, detecta el tipo de documento (Judgment, Directive, AG Opinion) y aplica el regex correspondiente:
- Judgment / AG Opinion: `^\d+\.\s` (párrafos numerados)
- Directive / Regulation: `^Article\s+\d+`
- Secciones romanas: `^[IVX]+\.\s+[A-Z]`
- Fallback: `RecursiveCharacterTextSplitter`

**Nota:** `parent_child` y `sliding_window` excluidos. Parent-child tiene problema de persistencia del docstore entre nodos Kedro; sliding_window es demasiado similar a fixed.

---

### FASE 2 — Embeddings (impacto alto · requiere re-indexar)

Chunking fijado al ganador de Fase 1. Retrieval fijado al baseline.
Hacer todos en bloque (cada modelo requiere reconstruir el índice FAISS completo).

| Experimento | Modelo | Notas |
|---|---|---|
| `exp2__gemini_emb2` | gemini-embedding-2 (= baseline) | Referencia |
| `exp2__text_emb3_small` | text-embedding-3-small (OpenAI) | Referencia de mercado |
| `exp2__multilingual_e5_large` | multilingual-e5-large (HuggingFace local) | Relevante: corpus EUR-Lex multilingüe |
| `exp2__bge_m3` | bge-m3 (HuggingFace local) | SOTA multilingüe, dense+sparse |

Métricas clave en este bloque: `ContextRecall` y `ContextPrecision` — indican si el retriever encuentra los chunks correctos con cada modelo.

---

### FASE 3 — Retrieval + Reranking

Chunking = ganador Fase 1. Embedding = ganador Fase 2.

| Experimento | Estrategia | Implementación |
|---|---|---|
| `exp3__dense_k3` | Dense cosine top-3 | `vectorstore.as_retriever(k=3)` |
| `exp3__dense_k5` | Dense cosine top-5 (= baseline) | `vectorstore.as_retriever(k=5)` |
| `exp3__dense_k10` | Dense cosine top-10 | `vectorstore.as_retriever(k=10)` |
| `exp3__mmr` | Maximal Marginal Relevance | `search_type="mmr", fetch_k=20, k=5` |
| `exp3__hybrid_bm25` | BM25 + Dense (RRF 50/50) | `EnsembleRetriever([BM25Retriever, vectorstore])` |
| `exp3__rerank_crossenc` | Dense k=50 → CrossEncoder top-5 | `ContextualCompressionRetriever + CrossEncoderReranker(bge-reranker-v2-m3)` |

**CrossEncoder ratio recomendado:** recuperar 50 candidatos → reranker selecciona top-5.
**Excluidos:** `SelfQueryRetriever` (incompatible con FAISS), `ContextualCompressionRetriever` con LLM extractor (latencia y coste).

---

### FASE 4 — RAG Avanzado (si da tiempo)

Sobre la mejor configuración de Fases 1-3.

| Experimento | Técnica | Implementación | Nota |
|---|---|---|---|
| `exp4__multi_query` | MultiQuery: LLM genera 3 variantes, fusiona resultados | `MultiQueryRetriever` (LangChain nativo) | Fácil |
| `exp4__hyde` | HyDE: LLM genera doc hipotético, se busca por su embedding | ~50 líneas custom sobre LangChain | Media |
| `exp4__contextual_ret` | Contextual Retrieval: añade 50-100 tokens de contexto a cada chunk antes de indexar | Nodo `enrich_chunks` antes de `build_index` | Media — actúa en indexación |

**Contextual Retrieval** requiere un nodo extra (`enrich_chunks_with_context`) que llama al LLM una vez por chunk antes de construir el índice. Es la única técnica de esta fase que modifica el pipeline de indexación.

**Excluidos:** `RAG-Fusion` (redundante con MultiQuery), `Step-back prompting` (poca ganancia en docs legales), `SelfQuery`.

---

### FASE 5 — Paradigmas alternativos (solo TFG texto)

Self-RAG, CRAG, GraphRAG: descripción teórica y justificación en la memoria, sin implementación. Su complejidad (LangGraph, Neo4j) supera el ROI para el alcance del TFG.

---

## 10. Tracking MLflow

Sin kedro-mlflow plugin (incompatible con Kedro 1.3.x). MLflow directamente en el nodo `evaluate_ragas`:

```python
import mlflow

def evaluate_ragas(rag_results, params):
    # ...evaluate...
    mlflow.set_experiment("rag_eval_eurlex")
    with mlflow.start_run(run_name=params["experiment_id"]):
        mlflow.log_params({
            "chunk_strategy": params["chunking"]["strategy"],
            "chunk_size":     params["chunking"].get("chunk_size"),
            "embedding_model": params["embedding"]["model"],
            "retrieval_strategy": params["retrieval"]["strategy"],
            "retrieval_top_k":   params["retrieval"]["top_k"],
        })
        mlflow.log_metrics({
            "context_precision":     scores["context_precision"],
            "context_recall":        scores["context_recall"],
            "context_entity_recall": scores["context_entity_recall"],
            "faithfulness":          scores["faithfulness"],
            "response_relevancy":    scores["response_relevancy"],
            "noise_sensitivity":     scores["noise_sensitivity"],
        })
    return results_df
```

`mlflow ui` → `localhost:5000` para comparar todos los experimentos.

---

## 11. Resumen de pipeline_registry final

```
__default__:      ingestion + corpus_selection + gold_dataset
ingestion:        ingestion
corpus_selection: corpus_selection
gold_dataset:     gold_dataset
exp0:             rag_base(exp0)
exp1__*:          rag_base(exp1__*)   ← 6 variantes chunking
exp2__*:          rag_base(exp2__*)   ← 4 variantes embedding
exp3__*:          rag_base(exp3__*)   ← 6 variantes retrieval
exp4__*:          rag_base(exp4__*)   ← 3 variantes avanzadas
```

---

## 12. Orden de ejecución recomendado

```
kedro run --pipeline exp0                    # Baseline
kedro run --pipeline exp1__fixed_256         # ┐
kedro run --pipeline exp1__fixed_1024        # │ Fase 1: Chunking
kedro run --pipeline exp1__recursive         # │ (actualizar ganador en exp2 params)
kedro run --pipeline exp1__structural        # │
kedro run --pipeline exp1__semantic          # ┘
kedro run --pipeline exp2__text_emb3_small   # ┐
kedro run --pipeline exp2__multilingual_e5   # │ Fase 2: Embeddings
kedro run --pipeline exp2__bge_m3            # ┘ (actualizar ganador en exp3 params)
kedro run --pipeline exp3__mmr               # ┐
kedro run --pipeline exp3__hybrid_bm25       # │ Fase 3: Retrieval
kedro run --pipeline exp3__rerank_crossenc   # ┘
kedro run --pipeline exp4__multi_query       # ┐
kedro run --pipeline exp4__hyde              # │ Fase 4: Avanzado (si da tiempo)
kedro run --pipeline exp4__contextual_ret    # ┘
```

---

## 13. Fuera de scope

- kedro-mlflow plugin (incompatible)
- Parent-child chunking (problema persistencia docstore)
- SelfQueryRetriever (incompatible con FAISS)
- RAG-Fusion, Step-back prompting (redundantes dado el scope)
- Self-RAG, CRAG, GraphRAG (demasiado complejos para TFG)
- Métricas Answer Coherence / Answer Fluency (no existen en RAGAS)
