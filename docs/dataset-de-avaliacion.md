# Conjunto de evaluación (gold dataset)

Decisiones de diseño del conjunto de pares pregunta–respuesta contra el que se miden todas las
configuraciones. Documenta **lo que se ejecutó**, no el plan inicial.

## 1. Del corpus fuente al corpus de trabajo

El corpus de partida es `gplsi/alia_intellectual_property` (EUR-Lex, dominio de propiedad
intelectual, CC BY 4.0), variante inglesa `eurlex-en-md.jsonl`: **40.181 documentos**.

Los experimentos **no** corren sobre esos 40.181 documentos. El pipeline `corpus_selection`
extrae un subcorpus temático de derecho de autor aplicando tres filtros:

| Filtro | Criterio |
|---|---|
| Palabra clave en el título | `copyright`, `related right`, `neighbouring right`, `computer program`, `software` (sin distinción de mayúsculas) |
| Tipo de acto jurídico (`form`) | Lista blanca de 16 formas: Judgment, Opinion of the Advocate General, Order, Abstract, Directive, Consolidated text, Decision, Regulation, Communication, Impact assessment, Green Paper, Opinion, Written question, Proposal for a directive, Proposal for a regulation, Amended proposal for a directive |
| Extensión mínima | 300 palabras |

Resultado: **363 documentos**. Los filtros están en el bloque `corpus_selection` de
`conf/base/parameters.yml`; cambiarlos es lo primero que hay que hacer para aplicar el marco a
otro dominio.

Distribución por longitud del subcorpus:

| Rango | Documentos | % |
|---|---|---|
| 300 – 1.000 palabras | 56 | 15,4 % |
| 1.000 – 5.000 palabras | 108 | 29,8 % |
| 5.000 – 20.000 palabras | 166 | 45,7 % |
| más de 20.000 palabras | 33 | 9,1 % |

## 2. Muestreo estratificado

De los 363 documentos se seleccionan **75** como material de generación.

El muestreo aleatorio simple habría sobrerrepresentado los documentos medianos (45,7 % del
corpus) e infrarrepresentado los largos (9,1 %), que son precisamente los más valiosos para
generar preguntas *multi-hop*: directivas y textos consolidados donde la respuesta exige
combinar varias secciones.

| Estrato | Pool | Seleccionados |
|---|---|---|
| 300 – 1.000 palabras | 56 | 10 |
| 1.000 – 5.000 palabras | 108 | 25 |
| 5.000 – 20.000 palabras | 166 | 25 |
| más de 20.000 palabras | 33 | 15 |
| **Total** | **363** | **75** |

Los documentos del estrato 4 son el 20 % de la muestra frente al 9,1 % del corpus:
sobrerrepresentación deliberada. `seed=42` fija el muestreo para que sea reproducible.

## 3. Generación de los pares pregunta–respuesta

**Herramienta:** RAGAS `TestsetGenerator`, vía `generate_with_langchain_docs()`.

Los 75 documentos entran como `Document` de LangChain. RAGAS aplica internamente un
`HeadlineSplitter` que los divide en secciones por encabezados, construye un *knowledge graph*
sobre los nodos resultantes y genera los pares a partir de él.

| Parámetro | Valor |
|---|---|
| Pares generados | **50** |
| LLM generador | `deepseek-v4-flash` (API compatible con OpenAI) |
| Embedder del knowledge graph | `gemini-embedding-2` |
| Idioma | Inglés (prompts por defecto de RAGAS) |
| Semilla | 42 |

**Distribución de tipos de pregunta**, configurada al 50 / 50 en
`gold_dataset.query_distribution`:

| Sintetizador | Pares | Qué produce |
|---|---|---|
| `single_hop_specific_query_synthesizer` | 26 | Pregunta factual respondible desde un solo fragmento |
| `multi_hop_specific_query_synthesizer` | 24 | Pregunta que combina hechos concretos de 2–3 fragmentos |

El sintetizador *multi-hop abstract*, presente en el plan inicial, se descartó: producía
preguntas demasiado vagas para evaluar recuperación con precisión.

## 4. Elección del embedder juez

`gemini-embedding-2` se usa tanto para construir el knowledge graph como para la métrica
`answer_relevancy`, y se mantiene **fijo en todos los experimentos** como variable de control.

La elección es deliberadamente neutral: ninguno de los embedders candidatos evaluados en la
fase 1 (`bge-m3`, `qwen3-8b`, `snowflake-arctic-l`, `bge-base`) pertenece a la familia Gemini,
de modo que el juez no favorece a ninguno de ellos. Por el mismo motivo el LLM juez de RAGAS
(`gemini-2.5-flash`) es distinto del LLM generador de respuestas (`deepseek-v4-flash`): evita
el sesgo de autoevaluación.

## 5. Reproducibilidad

- `seed=42` fija el muestreo estratificado.
- El conjunto se genera **una sola vez** y se versiona en `data/03_primary/gold_dataset.json`.
  No se regenera entre experimentos: todas las configuraciones responden exactamente a las
  mismas 50 preguntas.
- El *knowledge graph* intermedio se conserva en `data/03_primary/knowledge_graph.json`.

## 6. Coste

Generación del conjunto: aproximadamente 0,30 USD.

La evaluación completa es el grueso del gasto. Cada par pregunta–respuesta y cada variante
experimental consume alrededor de 12 llamadas al LLM juez (`faithfulness` sola requiere entre 4
y 6), de unos 1.000 tokens cada una. Con 50 preguntas y 29 ejecuciones, el orden de magnitud
está en unos pocos dólares.
