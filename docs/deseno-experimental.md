# Diseño experimental

Descripción de la barrida **tal como se ejecutó**. Al final se recogen las desviaciones
respecto al plan inicial.

## Principio: ablación secuencial con arrastre

Cada fase varía **un solo componente** y mantiene todo lo demás fijo (*ceteris paribus*). Al
terminar, un nodo `select_best_*` lee las métricas de todas las variantes, elige la ganadora
por `P+R` y la inyecta como configuración base de la fase siguiente.

Esto se ve en los propios ficheros de parámetros: los bloques de `exp2` **no declaran**
`embedding.model`. La clave ausente se resuelve en tiempo de ejecución al valor de
`best_embedding`. Cada fase hereda las decisiones ya tomadas sin que haya que reescribir nada a
mano.

**Métrica de decisión:** `P+R` = `context_precision` + `context_recall`. Se optimiza la
recuperación porque es lo que el pipeline controla directamente; las métricas de generación
(`faithfulness`, `answer_relevancy`) se vigilan como efecto secundario.

**Limitación asumida:** es una optimización *greedy*. Que A₁ gane en la primera fase no
garantiza que la combinación A₁+B₁ supere a A₂+B₂; el óptimo por pasos no tiene por qué ser el
óptimo global. Se eligió así porque una búsqueda exhaustiva sobre el espacio combinatorio
completo era inviable con el presupuesto de cómputo y de API del trabajo.

---

## Fase 0 — Baseline

El punto de referencia contra el que se mide todo lo demás. Deliberadamente simple:

| Componente | Valor |
|---|---|
| Chunking | `fixed_size`, 256 tokens, solape 25 (~10 %) |
| Embedding | `BAAI/bge-base-en-v1.5` (86M parámetros, 768 dim) |
| Retrieval | Coseno, `top_k=5` |
| LLM generador | `deepseek-v4-flash`, temperatura 0 |
| LLM juez | `gemini-2.5-flash`, temperatura 0 |

**Resultado: P+R = 0,756.**

El LLM generador y el juez se fijan aquí y **no cambian en ningún experimento posterior**: son
variables de control. Sin un baseline honesto no hay forma de saber si algo mejora o empeora.

---

## Fase 1 — Embeddings

Se varía únicamente el modelo de embedding. Chunking y retrieval quedan clavados al baseline.

| Variante | Modelo | P+R |
|---|---|---|
| *(= exp0)* | `BAAI/bge-base-en-v1.5` | 0,756 |
| `exp1__bge_m3` | **`BAAI/bge-m3`** | **0,999** |
| `exp1__qwen3_8b` | `Qwen/Qwen3-Embedding-8B` | 0,889 |
| `exp1__snowflake_arctic_l` | `Snowflake/snowflake-arctic-embed-l-v2.0` | 0,775 |

**Ganador: `bge-m3` (+0,242).** El salto individual más grande de toda la barrida.

Resultado notable: `Qwen3-Embedding-8B`, muy superior en el ranking MTEB y casi cien veces más
grande, queda por detrás de `bge-m3` en este corpus. Primer indicio de que las posiciones en
los rankings generalistas no se trasladan sin más a un dominio especializado.

`bge-base` no se repite como variante: es idéntico a exp0 y se reutiliza esa columna.

---

## Fase 2 — Chunking

Embedding fijado al ganador de la fase 1 (`bge-m3`, resuelto automáticamente).

| Variante | Estrategia | P+R |
|---|---|---|
| *(= exp1\_\_bge_m3)* | **`fixed_size` 256** | **0,999** |
| `exp2__sentence` | `sentence` 512 | 0,993 |
| `exp2__structural_pc` | `structural` + parent-child | 0,958 |
| `exp2__fixed_512` | `fixed_size` 512 | 0,913 |
| `exp2__semantic` | `semantic` (percentil) | 0,891 |
| `exp2__structural` | `structural` | 0,729 |

**Ganador: el chunking del baseline (fixed 256). Δ = 0,000.**

Ninguna de las cinco alternativas mejora el troceado de tamaño fijo más simple. El chunking
semántico, el más sofisticado del conjunto, queda quinto de seis. Los fragmentos cortos (256
frente a 512) funcionan mejor porque concentran la señal: en un corpus jurídico, un fragmento
largo mezcla varios preceptos y diluye la similitud con la consulta.

---

## Fase 3 — Retrieval

Embedding y chunking ya fijados. Esta fase **reutiliza el índice de la fase 2** en lugar de
reindexar, lo que abarata drásticamente la barrida.

| Variante | Estrategia | `top_k` | P+R |
|---|---|---|---|
| `exp3__mmr_k10` | **MMR (`fetch_k=30`)** | **10** | **1,094** |
| `exp3__dense_k10` | Coseno | 10 | 1,050 |
| `exp3__mmr` | MMR | 5 | 1,023 |
| *(= exp1\_\_bge_m3)* | Coseno | 5 | 0,999 |
| `exp3__dense_k3` | Coseno | 3 | 0,915 |
| `exp3__bm25_k10` | BM25 | 10 | 0,914 |
| `exp3__hybrid_bm25` | Híbrido denso + BM25 (RRF) | 5 | 0,898 |
| `exp3__hybrid_k10` | Híbrido denso + BM25 (RRF) | 10 | 0,886 |
| `exp3__bm25` | BM25 | 5 | 0,808 |

**Ganador: MMR con `top_k=10` (+0,095).**

Dos lecturas. Primera: ampliar `top_k` compensa. En las tres estrategias donde la recuperación
es puramente por ranking —MMR, coseno y BM25— pasar de 5 a 10 mejora el P+R: el recall gana más
de lo que la precisión pierde. La única excepción es el híbrido, donde ocurre lo contrario
(0,898 con k=5 frente a 0,886 con k=10).

Segunda: el retrieval híbrido, recomendación casi universal en la literatura, es de lo peor
aquí. BM25 aporta poco en un corpus donde la terminología está muy normalizada, y al fusionarlo
por RRF degrada el ranking denso en lugar de complementarlo.

---

## Fase 4 — Expansión de consulta

Siete técnicas, todas heredando embedding, chunking y retrieval ganadores.

| Variante | Técnica | P+R |
|---|---|---|
| *(= exp3\_\_mmr_k10)* | **Ninguna** | **1,094** |
| `exp4b__multiquery` | Multi-Query | 1,047 |
| `exp4c__rag_fusion` | RAG-Fusion | 1,015 |
| `exp4e__stepback` | Step-Back | 1,003 |
| `exp4d__rewrite` | Rewrite | 0,987 |
| `exp4f__decomposition` | Decomposition | 0,942 |
| `exp4a__hyde` | HyDE | 0,846 |
| `exp4g__selfquery` | Self-Query | 0,823 |

**Ganadora: ninguna. Δ = 0,000.**

Las siete empeoran el resultado, sin excepción. Es el hallazgo negativo más claro del trabajo.

La explicación probable está en el vocabulario: el corpus jurídico usa terminología literal y
muy normalizada («related right», «computer program»), y toda técnica de expansión reformula la
consulta con lenguaje del LLM, alejándola de esa literalidad. HyDE, que genera un documento
hipotético completo, es el que más daño hace — precisamente el más agresivo en la
reformulación.

---

## Fase 5 — Reranking

| Variante | Reranker | `top_k` final | P+R |
|---|---|---|---|
| `exp5b__rerank_ce_k10` | **Cross-encoder `BAAI/bge-reranker-v2-m3`** | **10** | **1,203** |
| `exp5a__rerank_ce_k5` | Cross-encoder `bge-reranker-v2-m3` | 5 | 1,187 |
| `exp5c__rerank_llm_k10` | LLM listwise (`deepseek-v4-flash`) | 10 | 1,184 |
| `exp5d__rerank_llm_k5` | LLM listwise (`deepseek-v4-flash`) | 5 | 1,154 |
| *(= exp3\_\_mmr_k10)* | Sin reranking | 10 | 1,094 |

**Ganador: cross-encoder con `top_k=10` (+0,109).**

El reranking opera en cascada: pool de 50 candidatos → 20 seleccionados por MMR → reranker →
top-k final. El cross-encoder, mucho más barato que el reranker por LLM, lo supera en las
cuatro comparaciones.

---

## Control: el stack «SOTA»

Ejecutado aparte, sin arrastre: combina la técnica **más avanzada sobre el papel** de cada
dimensión, no la ganadora empírica.

| Dimensión | Stack SOTA | Configuración empírica |
|---|---|---|
| Embedding | `Qwen3-Embedding-8B` (top MTEB) | `bge-m3` |
| Chunking | `semantic` | `fixed_size` 256 |
| Retrieval | Híbrido denso + BM25 (RRF) | MMR `k=10` |
| Expansión | RAG-Fusion | Ninguna |
| Reranking | LLM listwise (RankGPT) | Cross-encoder `bge-reranker-v2-m3` |

| Configuración | P+R |
|---|---|
| RAG naive (exp0) | 0,756 |
| Stack SOTA (expSOTA) | 1,095 |
| **Configuración empírica (exp5b)** | **1,203** |

El stack SOTA mejora el baseline, pero **queda un 9 % por debajo** de la configuración hallada
midiendo, y es sensiblemente más caro de ejecutar (embedding de 8B, expansión de consulta con
LLM y reranking generativo).

Este contraste es la justificación del marco entero: apilar recomendaciones de la literatura no
equivale a optimizar. Hay que medir sobre el corpus propio.

---

## Desviaciones respecto al plan inicial

| Plan inicial | Ejecutado | Motivo |
|---|---|---|
| 7 métricas RAGAS + latencia | **6 métricas**: `context_recall`, `context_precision`, `context_entity_recall`, `faithfulness`, `answer_relevancy`, `noise_sensitivity` | `factual_correctness` y `semantic_similarity` se descartaron; la latencia quedó fuera de alcance |
| Baseline con `bge-small`, fixed 512 | `bge-base-en-v1.5`, fixed 256 | Un baseline más razonable hace la comparación más exigente y por tanto más honesta |
| Embeddings multilingües (`multilingual-e5`, `paraphrase-multilingual`) | `bge-m3`, `qwen3-8b`, `snowflake-arctic-l` | El corpus es monolingüe en inglés; se priorizaron modelos punteros en inglés |
| Reranking dentro de la fase 4 | Fase 5 propia, separada de la expansión de consulta | Son etapas distintas del pipeline (pre y post recuperación) y mezclarlas rompía el *ceteris paribus* |
| Fase 4b: CRAG, Adaptive-RAG, RAG agéntico, GraphRAG | **No ejecutada** | Fuera del presupuesto de tiempo; queda como trabajo futuro |
| — | **expSOTA** (control frente a la literatura) | Añadido durante la ejecución; acabó siendo el resultado más relevante del trabajo |

El plan original está descrito en la memoria; estas desviaciones se documentan porque el
recorrido —incluidas las fases que no aportaron nada— es parte del resultado.
