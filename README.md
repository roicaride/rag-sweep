# Optimización experimental de sistemas RAG

**Un marco reproducible para encontrar la configuración RAG óptima de un dominio concreto — y su aplicación a un corpus jurídico europeo.**

Trabajo de Fin de Grao · Grao en Empresa e Tecnoloxía · Universidade de Santiago de Compostela
Autor: Roi Caride Borrajo · Memoria en galego · 2026

---

## Qué es esto

No existe una configuración RAG universalmente óptima. Lo que funciona en un corpus de
documentación técnica no tiene por qué funcionar en uno jurídico, médico o financiero, y la
única forma honesta de saberlo es **medirlo sobre el corpus propio**.

Este repositorio contiene **la herramienta que hace esa medición sistemática y repetible**: un
pipeline de ablación secuencial construido sobre Kedro que compara variantes de embedding,
chunking, retrieval, expansión de consulta y reranking, evalúa cada una con las mismas métricas
sobre el mismo conjunto de preguntas, y **encadena automáticamente el ganador de cada fase como
punto de partida de la siguiente**.

> **Los resultados que verás aquí son un caso de aplicación, no el producto.**
> Se ejecutó sobre un corpus de derecho de autor de EUR-Lex porque había que ejecutarlo sobre
> algo. Los números concretos (que `bge-m3` gane, que MMR con k=10 gane, que ninguna técnica de
> expansión de consulta ayude) **son propios de ese corpus y no deben extrapolarse**. Lo que sí
> es transferible es el procedimiento: cambia el corpus y los ficheros de parámetros, vuelve a
> ejecutar, y obtendrás la configuración óptima *de tu dominio* con la misma evidencia detrás.

---

## La evidencia de que la herramienta hace falta

El experimento más relevante del trabajo no es cuál fue la mejor configuración, sino esta
comparación:

| Configuración | P+R |
|---|---|
| RAG naive (baseline) | 0,756 |
| **Stack «SOTA» montado desde la literatura** | **1,095** |
| **Configuración hallada empíricamente en este corpus** | **1,203** |

Un stack construido apilando las técnicas que los papers reportan como estado del arte
(chunking semántico + embedding de 8B parámetros + retrieval híbrido + RAG-Fusion + reranking
por LLM) **rinde un 9 % peor** que la configuración encontrada midiendo sobre el corpus real, y
además es mucho más caro de ejecutar.

Esa brecha es el argumento del proyecto: seguir las recomendaciones genéricas de la literatura
no equivale a optimizar. Hace falta un método, y el método hay que instrumentarlo.

---

## Cómo funciona

**Ablación secuencial con arrastre de ganadores.** Cada fase varía un único componente y deja
todo lo demás fijo. Al terminar, un nodo de selección elige el ganador por la métrica de
decisión y lo inyecta como configuración base de la fase siguiente.

```
exp0  baseline           →  fixed 256 · bge-base-en-v1.5 · coseno k=5
  ↓  select_best_embedding
exp1  embeddings         →  bge-m3 vs qwen3-8b vs snowflake-arctic-l
  ↓  select_best_chunking
exp2  chunking           →  fixed 512 · sentence · structural · structural+parent-child · semantic
  ↓  select_best_retrieval
exp3  retrieval          →  denso k3/k10 · MMR k5/k10 · BM25 k5/k10 · híbrido k5/k10
  ↓  select_best_query_transform
exp4  expansión consulta →  HyDE · Multi-Query · RAG-Fusion · Rewrite · Step-Back · Decomposition · Self-Query
  ↓  select_best_rerank
exp5  reranking          →  cross-encoder k5/k10 · LLM-as-reranker k5/k10

expSOTA  control: stack maximalista de la literatura, ejecutado aparte para contrastar
```

El arrastre está implementado en [`pipeline_registry.py`](rag-eval/src/rag_eval/pipeline_registry.py):
los nodos `select_best_*` leen las métricas de todas las variantes de una fase y producen un
dataset (`best_embedding`, `best_chunking`, …) que las fases posteriores reciben como entrada.
Añadir una variante nueva es **añadir un bloque a un YAML**, no tocar código.

Las fases 3, 4 y 5 reutilizan el índice vectorial de la fase 2 en lugar de reindexar, lo que
reduce el coste de una barrida completa de horas a minutos.

**Limitación conocida y asumida:** una optimización greedy por fases no garantiza el óptimo
global — que A₁ gane en el primer paso no implica que la combinación A₁+B₁ supere a A₂+B₂. Es
una búsqueda secuencial de hiperparámetros, elegida porque una búsqueda exhaustiva sobre el
espacio completo era inviable con el presupuesto del trabajo.

---

## Métricas

Seis métricas de RAGAS, con un LLM juez fijo (`gemini-2.5-flash`) e independiente del LLM
generador (`deepseek-v4-flash`) para evitar sesgo de autoevaluación:

| Métrica | Qué mide | Etapa |
|---|---|---|
| `context_recall` | ¿Se recuperó todo lo necesario para responder? | Recuperación |
| `context_precision` | ¿Lo recuperado es relevante y está bien ordenado? | Recuperación |
| `context_entity_recall` | Cobertura de las entidades clave de la respuesta ideal | Recuperación |
| `faithfulness` | ¿La respuesta se sostiene en el contexto recuperado? | Generación |
| `answer_relevancy` | ¿La respuesta responde a la pregunta formulada? | Generación |
| `noise_sensitivity` | ¿Introduce errores cuando hay contexto irrelevante? | Robustez |

**Métrica de decisión: `P+R` = `context_precision` + `context_recall`.** Se optimiza la etapa de
recuperación porque es la que el pipeline controla; la calidad de generación se vigila como
efecto secundario, no como objetivo.

---

## Estructura del repositorio

```
.
├── rag-eval/            Pipeline Kedro — el núcleo reutilizable
│   ├── conf/base/       parameters_exp*.yml → una barrida = un fichero de configuración
│   ├── src/rag_eval/    5 pipelines + módulos de chunking, embedding, retrieval, reranking
│   ├── notebooks/       exploración del corpus, construcción del gold set, análisis final
│   └── info/            notebooks precursores (prototipo previo al pipeline)
├── resultados/          Caso de ejemplo EUR-Lex: 22 figuras + 10 tablas CSV
├── memoria/             Memoria del TFG: fuente LaTeX, bibliografía, figuras y PDF final
└── docs/
    ├── deseno-experimental.md      Diseño ejecutado, fase a fase
    ├── dataset-de-avaliacion.md    Construcción del conjunto de evaluación
    └── specs/                      Decisiones técnicas documentadas durante el desarrollo
```

Lo que **no** está en el repositorio, por peso y por ser regenerable: el corpus bruto
(3,1 GB), el corpus limpio (2,1 GB), los índices Qdrant (5,8 GB) y la base de MLflow. El
pipeline los reconstruye desde la fuente pública.

---

## El caso de aplicación: EUR-Lex

| | |
|---|---|
| Corpus fuente | `gplsi/alia_intellectual_property` (EUR-Lex, propiedad intelectual, CC BY 4.0) — 40.181 documentos |
| Subcorpus de trabajo | 363 documentos de derecho de autor, filtrados por palabra clave en el título, tipo de acto jurídico y mínimo de 300 palabras |
| Conjunto de evaluación | 50 pares pregunta–respuesta generados con RAGAS (26 single-hop, 24 multi-hop), muestreo estratificado por longitud, `seed=42` |
| Configuraciones evaluadas | 29 ejecuciones completas a lo largo de 6 fases |
| Vector store | Qdrant en modo servidor |
| Orquestación / tracking | Kedro 1.3.1 · MLflow |

### Progresión del pipeline

| Fase | Decisión | P+R | Δ |
|---|---|---|---|
| Baseline | fixed 256 · bge-base-en-v1.5 · coseno k=5 | 0,756 | — |
| + Embedding | **bge-m3** | 0,999 | +0,242 |
| + Chunking | fixed 256 *(gana el baseline)* | 0,999 | 0,000 |
| + Retrieval | **MMR k=10** | 1,094 | +0,095 |
| + Expansión | **ninguna** *(las 7 técnicas empeoran)* | 1,094 | 0,000 |
| + Reranking | **cross-encoder `bge-reranker-v2-m3` k=10** | **1,203** | +0,109 |

**+59 % sobre el baseline.** Dos de las cinco fases no aportaron nada: el chunking del baseline
resultó ser ya el mejor, y **las siete técnicas de expansión de consulta empeoraron el
resultado sin excepción** — un hallazgo negativo que probablemente se explica por lo específico
del vocabulario jurídico, donde reescribir la consulta la aleja de la terminología literal del
corpus.

Todas las cifras salen de [`resultados/tablas/`](resultados/tablas/) y son las mismas que se
reportan en la memoria.

---

## Reutilizarlo en otro dominio

1. Sustituye el corpus en `rag-eval/data/01_raw/` y ajusta el bloque `corpus_selection` de
   `conf/base/parameters.yml` con los filtros de tu dominio.
2. Genera el conjunto de evaluación: `kedro run --pipeline gold_dataset`.
3. Lanza el baseline y las fases que te interesen. Las variantes se declaran en
   `conf/base/parameters_exp*.yml`; para probar un embedding nuevo basta con copiar un bloque y
   cambiar el nombre del modelo.
4. `notebooks/03_results.ipynb` regenera todas las figuras y tablas comparativas.

No hace falta tocar código Python salvo que quieras añadir una **estrategia** que no exista
(un tipo de chunking nuevo, por ejemplo); en ese caso el punto de extensión es el módulo
correspondiente en `src/rag_eval/`.

Instrucciones detalladas de instalación y ejecución: [`rag-eval/README.md`](rag-eval/README.md).

---

## Alcance y honestidad

Esto es un **activo experimental reutilizable**, no un producto desplegado. Concretamente:

- **No hay servicio, API ni interfaz de usuario.** El pipeline produce evidencia, no un sistema
  en producción.
- **No hay medición de latencia ni de coste en producción.** Se descartaron del alcance.
- **No se evaluaron arquitecturas alternativas** (CRAG, Adaptive-RAG, RAG agéntico, GraphRAG).
  Estaban en el plan inicial y quedaron en trabajo futuro.
- **Los resultados valen para este corpus.** Ese es, precisamente, el argumento del trabajo.

---

## Memoria

[`memoria/TFG_RAG_Roi_Caride.pdf`](memoria/TFG_RAG_Roi_Caride.pdf) — 48 páginas, en galego.
Fuente LaTeX incluida (`TFG_RAG.tex` + `referencias.bib`); compilar con
`pdflatex → biber → pdflatex → pdflatex`.

## Créditos

Corpus derivado de [`gplsi/alia_intellectual_property`](https://huggingface.co/datasets/gplsi/alia_intellectual_property)
(Espinosa Zaragoza et al., 2025), CC BY 4.0. Construido con
[Kedro](https://kedro.org), [LangChain](https://www.langchain.com),
[RAGAS](https://docs.ragas.io), [Qdrant](https://qdrant.tech) y [MLflow](https://mlflow.org).
