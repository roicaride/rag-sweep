# Resultados — caso de aplicación EUR-Lex

> **Estas cifras son un ejemplo de uso del marco, no su producto.**
> Corresponden a un corpus concreto: 363 documentos de derecho de autor de EUR-Lex, en inglés,
> evaluados sobre 50 pares pregunta–respuesta. **Los ganadores de cada fase valen para ese
> corpus y no deben extrapolarse a otro dominio** — precisamente esa es la tesis del trabajo.
> Lo reutilizable es el pipeline que produjo estos números: ver [`../README.md`](../README.md).

Todo el contenido de esta carpeta lo regenera
[`rag-eval/notebooks/03_results.ipynb`](../rag-eval/notebooks/03_results.ipynb) a partir de los
ficheros `eval_metrics.json` de cada experimento.

## Tablas

| Fichero | Contenido |
|---|---|
| `master_results.csv` | Las 24 configuraciones comparables, 6 métricas + P+R, ordenadas por P+R |
| `progression.csv` | Ganancia acumulada fase a fase, desde el baseline hasta el pipeline final |
| `final_metrics.csv` | Baseline frente a pipeline final, métrica a métrica |
| `sota_stack_vs_empirical.csv` | Naive · stack SOTA de la literatura · configuración empírica |
| `metrics_by_decision.csv` | Evolución de las 6 métricas a lo largo de las decisiones |
| `exp1_embeddings.csv` | Fase 1 — comparativa de modelos de embedding |
| `exp2_chunking.csv` | Fase 2 — comparativa de estrategias de troceado |
| `exp3_retrieval.csv` | Fase 3 — comparativa de estrategias de recuperación |
| `exp4_query_expansion.csv` | Fase 4 — las 7 técnicas de expansión de consulta |
| `exp5_reranking.csv` | Fase 5 — cross-encoder frente a reranker por LLM |

## Figuras

Cada figura está en PNG (para leer aquí) y en PDF vectorial (el que usa la memoria LaTeX).

| Fichero | Qué muestra |
|---|---|
| `progression` | Progresión del P+R fase a fase — la vista de conjunto |
| `final_metrics` | Baseline frente a pipeline final en las 6 métricas |
| `sota_stack_vs_empirical` | Stack de la literatura frente a configuración empírica |
| `metrics_evolution` | Trayectoria de cada métrica a lo largo de la barrida |
| `metrics_delta_heatmap` | Mapa de calor: qué métrica movió cada decisión, y en qué dirección |
| `exp1_embeddings` … `exp5_reranking` | Comparativa interna de cada fase |
| `exp4_recall_precision` | Dispersión precisión–recall de las técnicas de expansión |

## Lectura rápida

| | P+R |
|---|---|
| Baseline (RAG naive) | 0,756 |
| Stack «SOTA» de la literatura | 1,095 |
| **Configuración hallada empíricamente** | **1,203** |

La configuración ganadora en este corpus: `bge-m3` · chunking fijo de 256 tokens · MMR con
k=10 · sin expansión de consulta · reranking con cross-encoder `bge-reranker-v2-m3` y k=10.

Desglose completo del recorrido, incluidas las dos fases que no aportaron nada:
[`../docs/deseno-experimental.md`](../docs/deseno-experimental.md).
