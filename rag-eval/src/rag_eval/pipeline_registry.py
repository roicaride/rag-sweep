from kedro.pipeline import node, pipeline

from rag_eval.pipelines.ingestion.pipeline import create_pipeline as ingestion_pipeline
from rag_eval.pipelines.corpus_selection.pipeline import create_pipeline as corpus_selection_pipeline
from rag_eval.pipelines.gold_dataset.pipeline import create_pipeline as gold_pipeline
from rag_eval.pipelines.rag_base.pipeline import create_pipeline as rag_base_pipeline
from rag_eval.pipelines.evaluation.pipeline import create_pipeline as evaluation_pipeline
from rag_eval.pipelines.rag_base.nodes import resolve_params
from rag_eval.selection import select_best_embedding, select_best_chunking, select_best_retrieval, select_best_rerank, select_best_query_transform


def _experiment(name: str, carry=None):
    return rag_base_pipeline(name, carry=carry) + evaluation_pipeline(name, carry=carry)


def _retrieval_experiment(name: str, carry=("embedding", "chunking")):
    """exp3/exp4: retrieval-only sobre o índice compartido (non re-indexa)."""
    carry = list(carry)
    inputs = [f"params:{name}", "best_embedding"]
    if "chunking" in carry:
        inputs.append("best_chunking")
    if "retrieval" in carry:
        inputs.append("best_retrieval")
    resolve = pipeline([node(
        func=resolve_params,
        inputs=inputs,
        outputs=f"{name}.params",
        name=f"{name}_resolve_params",
    )])
    return resolve + evaluation_pipeline(
        name, carry=carry, source="params:retrieval_index_collection",
    )


def register_pipelines() -> dict:
    ingestion        = ingestion_pipeline()
    corpus_selection = corpus_selection_pipeline()
    gold             = gold_pipeline()

    # Selección do embedder gañador de exp1 (recall→precision) → best_embedding
    select = pipeline([node(
        func=select_best_embedding,
        inputs=["exp0.eval_metrics", "exp1__snowflake_arctic_l.eval_metrics",
                "exp1__bge_m3.eval_metrics", "exp1__qwen3_8b.eval_metrics",
                "params:exp0", "params:exp1__snowflake_arctic_l",
                "params:exp1__bge_m3", "params:exp1__qwen3_8b"],
        outputs="best_embedding",
        name="select_best_embedding",
    )])

    # Selección do chunking gañador de exp2 (incl. 256 = exp1__bge_m3) → best_chunking
    select_chunk = pipeline([node(
        func=select_best_chunking,
        inputs=["exp1__bge_m3.eval_metrics", "exp2__fixed_512.eval_metrics",
                "exp2__sentence.eval_metrics", "exp2__structural.eval_metrics",
                "exp2__structural_pc.eval_metrics", "exp2__semantic.eval_metrics",
                "params:exp1__bge_m3", "params:exp2__fixed_512",
                "params:exp2__sentence", "params:exp2__structural",
                "params:exp2__structural_pc", "params:exp2__semantic"],
        outputs="best_chunking",
        name="select_best_chunking",
    )])

    # Selección do retrieval gañador de exp3 (incl. cosine k5 = exp1__bge_m3) → best_retrieval
    _r = ["exp1__bge_m3", "exp3__dense_k10",
          "exp3__mmr_k5", "exp3__mmr_k10", "exp3__bm25_k5", "exp3__bm25_k10",
          "exp3__hybrid_k5", "exp3__hybrid_k10"]
    select_retr = pipeline([node(
        func=select_best_retrieval,
        inputs=[f"{e}.eval_metrics" for e in _r] + [f"params:{e}" for e in _r],
        outputs="best_retrieval",
        name="select_best_retrieval",
    )])

    # Selección do reranking gañador de exp5 (incl. baseline sen rerank = mmr_k10) → best_rerank
    _rr = ["exp5a__rerank_ce_k5", "exp5b__rerank_ce_k10", "exp5c__rerank_llm_k5",
           "exp5d__rerank_llm_k10", "exp3__mmr_k10"]
    select_rerank = pipeline([node(
        func=select_best_rerank,
        inputs=[f"{e}.eval_metrics" for e in _rr] + [f"params:{e}" for e in _rr],
        outputs="best_rerank",
        name="select_best_rerank",
    )])

    # Selección da query-expansion gañadora de exp4a–4g (vs baseline mmr_k10 e
    # campión rerank ce_k10) → best_query_transform
    _qt = ["exp4a__hyde", "exp4b__multiquery", "exp4c__rag_fusion", "exp4d__rewrite",
           "exp4e__stepback", "exp4f__decomposition", "exp4g__selfquery",
           "exp3__mmr_k10", "exp5b__rerank_ce_k10"]
    select_qt = pipeline([node(
        func=select_best_query_transform,
        inputs=[f"{e}.eval_metrics" for e in _qt] + [f"params:{e}" for e in _qt],
        outputs="best_query_transform",
        name="select_best_query_transform",
    )])

    return {
        "__default__":      ingestion + corpus_selection + gold,
        "ingestion":        ingestion,
        "corpus_selection": corpus_selection,
        "gold_dataset":     gold,
        # exp0 — baseline; exp1 — embeddings (bge-base = exp0)
        "exp0":                     _experiment("exp0"),
        "exp1__snowflake_arctic_l": _experiment("exp1__snowflake_arctic_l"),
        "exp1__bge_m3":             _experiment("exp1__bge_m3"),
        "exp1__qwen3_8b":           _experiment("exp1__qwen3_8b"),
        # selección de gañadores (arrastre)
        "select_best_embedding":    select,
        "select_best_chunking":     select_chunk,
        "select_best_retrieval":    select_retr,
        "select_best_rerank":       select_rerank,
        "select_best_query_transform": select_qt,
        # exp2 — chunking (embedding = gañador exp1, auto)
        "exp2__fixed_512":     _experiment("exp2__fixed_512", carry=["embedding"]),
        "exp2__sentence":      _experiment("exp2__sentence", carry=["embedding"]),
        "exp2__structural":    _experiment("exp2__structural", carry=["embedding"]),
        "exp2__structural_pc": _experiment("exp2__structural_pc", carry=["embedding"]),
        "exp2__semantic":      _experiment("exp2__semantic", carry=["embedding"]),
        # exp3 — retrieval (reutiliza índice 256+bge-m3, non re-indexa). Agrupado por estratexia.
        "exp3__dense_k10":     _retrieval_experiment("exp3__dense_k10"),
        "exp3__mmr_k5":        _retrieval_experiment("exp3__mmr_k5"),
        "exp3__mmr_k10":       _retrieval_experiment("exp3__mmr_k10"),
        "exp3__bm25_k5":       _retrieval_experiment("exp3__bm25_k5"),
        "exp3__bm25_k10":      _retrieval_experiment("exp3__bm25_k10"),
        "exp3__hybrid_k5":     _retrieval_experiment("exp3__hybrid_k5"),
        "exp3__hybrid_k10":    _retrieval_experiment("exp3__hybrid_k10"),
        # exp4a–4g — query expansion (herda embedding+chunking+retrieval, reutiliza índice)
        "exp4a__hyde":          _retrieval_experiment("exp4a__hyde", carry=["embedding", "chunking", "retrieval"]),
        "exp4b__multiquery":    _retrieval_experiment("exp4b__multiquery", carry=["embedding", "chunking", "retrieval"]),
        "exp4c__rag_fusion":    _retrieval_experiment("exp4c__rag_fusion", carry=["embedding", "chunking", "retrieval"]),
        "exp4d__rewrite":       _retrieval_experiment("exp4d__rewrite", carry=["embedding", "chunking", "retrieval"]),
        "exp4e__stepback":      _retrieval_experiment("exp4e__stepback", carry=["embedding", "chunking", "retrieval"]),
        "exp4f__decomposition": _retrieval_experiment("exp4f__decomposition", carry=["embedding", "chunking", "retrieval"]),
        "exp4g__selfquery":     _retrieval_experiment("exp4g__selfquery", carry=["embedding", "chunking", "retrieval"]),
        # exp5a–5d — reranking (herda embedding+chunking+retrieval, reutiliza índice)
        "exp5a__rerank_ce_k5":   _retrieval_experiment("exp5a__rerank_ce_k5", carry=["embedding", "chunking", "retrieval"]),
        "exp5b__rerank_ce_k10":  _retrieval_experiment("exp5b__rerank_ce_k10", carry=["embedding", "chunking", "retrieval"]),
        "exp5c__rerank_llm_k5":  _retrieval_experiment("exp5c__rerank_llm_k5", carry=["embedding", "chunking", "retrieval"]),
        "exp5d__rerank_llm_k10": _retrieval_experiment("exp5d__rerank_llm_k10", carry=["embedding", "chunking", "retrieval"]),
        # expSOTA — stack SOTA maximalista (re-indexa semantic+qwen3, logo eval con
        # rag_fusion sobre hybrid + LLM rerank). Experimento completo, sen carry.
        "expSOTA": _experiment("expSOTA"),
    }
