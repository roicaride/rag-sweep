# Evaluation dataset (gold dataset)

Design decisions behind the set of question–answer pairs every configuration is measured
against. This documents **what was executed**, not the initial plan.

## 1. From source corpus to working corpus

The starting corpus is `gplsi/alia_intellectual_property` (EUR-Lex, intellectual property
domain, CC BY 4.0), English variant `eurlex-en-md.jsonl`: **40,181 documents**.

The experiments do **not** run on those 40,181 documents. The `corpus_selection` pipeline
extracts a thematic copyright subcorpus by applying three filters:

| Filter | Criterion |
|---|---|
| Title keyword | `copyright`, `related right`, `neighbouring right`, `computer program`, `software` (case-insensitive) |
| Legal act type (`form`) | Allowlist of 16 forms: Judgment, Opinion of the Advocate General, Order, Abstract, Directive, Consolidated text, Decision, Regulation, Communication, Impact assessment, Green Paper, Opinion, Written question, Proposal for a directive, Proposal for a regulation, Amended proposal for a directive |
| Minimum length | 300 words |

Result: **363 documents**. The filters live in the `corpus_selection` block of
`conf/base/parameters.yml`; changing them is the first thing to do when applying the framework
to another domain.

Length distribution of the subcorpus:

| Range | Documents | % |
|---|---|---|
| 300 – 1,000 words | 56 | 15.4% |
| 1,000 – 5,000 words | 108 | 29.8% |
| 5,000 – 20,000 words | 166 | 45.7% |
| over 20,000 words | 33 | 9.1% |

## 2. Stratified sampling

**75** of the 363 documents are selected as generation material.

Simple random sampling would have over-represented medium-length documents (45.7% of the
corpus) and under-represented long ones (9.1%) — precisely the most valuable for generating
*multi-hop* questions: directives and consolidated texts where answering requires combining
several sections.

| Stratum | Pool | Selected |
|---|---|---|
| 300 – 1,000 words | 56 | 10 |
| 1,000 – 5,000 words | 108 | 25 |
| 5,000 – 20,000 words | 166 | 25 |
| over 20,000 words | 33 | 15 |
| **Total** | **363** | **75** |

Stratum 4 documents make up 20% of the sample against 9.1% of the corpus: deliberate
over-representation. `seed=42` fixes the sampling so it stays reproducible.

## 3. Generating the question–answer pairs

**Tool:** RAGAS `TestsetGenerator`, via `generate_with_langchain_docs()`.

The 75 documents go in as LangChain `Document`s. RAGAS internally applies a `HeadlineSplitter`
that divides them into sections by heading, builds a knowledge graph over the resulting nodes,
and generates the pairs from it.

| Parameter | Value |
|---|---|
| Pairs generated | **50** |
| Generator LLM | `deepseek-v4-flash` (OpenAI-compatible API) |
| Knowledge graph embedder | `gemini-embedding-2` |
| Language | English (RAGAS default prompts) |
| Seed | 42 |

**Question type distribution**, configured 50/50 in `gold_dataset.query_distribution`:

| Synthesizer | Pairs | What it produces |
|---|---|---|
| `single_hop_specific_query_synthesizer` | 26 | Factual question answerable from a single chunk |
| `multi_hop_specific_query_synthesizer` | 24 | Question combining concrete facts from 2–3 chunks |

The *multi-hop abstract* synthesizer, present in the initial plan, was dropped: it produced
questions too vague to evaluate retrieval precisely.

## 4. Choosing the judge embedder

`gemini-embedding-2` is used both to build the knowledge graph and for the `answer_relevancy`
metric, and is held **fixed across every experiment** as a control variable.

The choice is deliberately neutral: none of the candidate embedders evaluated in stage 1
(`bge-m3`, `qwen3-8b`, `snowflake-arctic-l`, `bge-base`) belongs to the Gemini family, so the
judge favours none of them. For the same reason the RAGAS judge LLM (`gemini-2.5-flash`) is
different from the answer generator LLM (`deepseek-v4-flash`): it avoids self-evaluation bias.

## 5. Reproducibility

- `seed=42` fixes the stratified sampling.
- The set is generated **once** and versioned at `data/03_primary/gold_dataset.json`. It is
  never regenerated between experiments: every configuration answers exactly the same 50
  questions.
- The intermediate knowledge graph is kept at `data/03_primary/knowledge_graph.json`.

## 6. Cost

Generating the set: roughly USD 0.30.

Evaluation is where the spending goes. Each question–answer pair and each experimental variant
consumes around 12 judge-LLM calls (`faithfulness` alone needs 4 to 6), of about 1,000 tokens
each. With 50 questions and 29 runs, the order of magnitude is a few dollars.
