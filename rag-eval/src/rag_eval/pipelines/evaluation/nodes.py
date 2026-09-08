import os
import re
import time

from rag_eval.retrieval import get_retriever, make_reranker, candidate_retriever
from rag_eval.query_transform import query_transform_retriever


def _retry(fn, label: str, tries: int = 8, wait: int = 12):
    """Reintenta ante erros transitorios da API (504/500/429/timeout)."""
    for t in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = any(s in msg for s in (
                "500", "502", "503", "504", "gateway", "time-out", "timeout",
                "429", "resource_exhausted", "connection"))
            if transient and t < tries - 1:
                print(f"  {label}: erro transitorio (agardo {wait}s, {t+1}/{tries}): {str(e)[:80]}")
                time.sleep(wait)
            else:
                raise


def run_rag(collection_name: str, gold_dataset: list[dict], params: dict, rag_prompt: str, qdrant_url: str) -> list[dict]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from rag_eval.embedding import get_embeddings

    embeddings    = get_embeddings(params["embedding"]["model"])
    client        = QdrantClient(url=qdrant_url)
    vectorstore   = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

    llm_cfg   = params["llm"]
    rcfg      = params["retrieval"]
    if "query_transform" in params and "rerank" in params:
        qt, rr = params["query_transform"], params["rerank"]
        base = query_transform_retriever(vectorstore, rcfg, qt, out_k=rr["fetch_k"])
        retriever = make_reranker(base, rr)
        print(f"run_rag: query-transform {qt['type']} ({rcfg['strategy']}) -> pool {rr['fetch_k']} -> rerank {rr['type']} -> top_k={rr['top_k']}")
    elif "rerank" in params:
        rr = params["rerank"]
        base = candidate_retriever(vectorstore, rcfg, rr["fetch_k"], rr["pool"])
        retriever = make_reranker(base, rr)
        print(f"run_rag: cascada {rcfg['strategy']} pool={rr['pool']} -> {rr['fetch_k']} -> rerank {rr['type']} -> top_k={rr['top_k']}")
    elif "query_transform" in params:
        qt = params["query_transform"]
        retriever = query_transform_retriever(vectorstore, rcfg, qt)
        print(f"run_rag: query-transform {qt['type']} sobre {rcfg['strategy']} top_k={rcfg['top_k']}")
    else:
        retriever = get_retriever(vectorstore, rcfg["strategy"], rcfg)
    llm       = ChatOpenAI(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        base_url=llm_cfg["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        timeout=60, max_retries=1,
    )
    chain = ChatPromptTemplate.from_template(rag_prompt) | llm | StrOutputParser()

    results = []
    total   = len(gold_dataset)
    for i, sample in enumerate(gold_dataset):
        query              = sample["user_input"]
        docs               = _retry(lambda: retriever.invoke(query), f"retrieval q{i+1}")
        # parent-child: se o chunk trae parent_text no payload, úsase o parent como contexto
        retrieved_contexts = [d.metadata.get("parent_text") or d.page_content for d in docs]
        response           = _retry(lambda: chain.invoke({
            "context":  "\n\n---\n\n".join(retrieved_contexts),
            "question": query,
        }), f"generation q{i+1}")
        results.append({
            "user_input":         query,
            "retrieved_contexts": retrieved_contexts,
            "response":           response,
            "reference":          sample["reference"],
        })
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  run_rag: {i + 1}/{total} queries procesadas")

    print(f"run_rag: {len(results)} resultados xerados")
    return results


def evaluate_ragas(rag_results: list[dict], params: dict, mlflow_experiment: str, ragas_run_config: dict, ragas_embedding: str) -> dict:
    import mlflow
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        ContextPrecision, ContextRecall, ContextEntityRecall,
        Faithfulness, ResponseRelevancy, NoiseSensitivity,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm_cfg   = params["ragas_llm"]
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
            max_output_tokens=llm_cfg["max_tokens"],
            thinking_budget=llm_cfg["thinking_budget"],  # 0: o thinking trunca o JSON estruturado do xuíz
            google_api_key=os.environ["GOOGLE_API_KEY"],
        ),
    )
    judge_emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=ragas_embedding, google_api_key=os.environ["GOOGLE_API_KEY"])
    )

    dataset = EvaluationDataset.from_list(rag_results)
    metrics = [
        ContextPrecision(), ContextRecall(), ContextEntityRecall(),
        Faithfulness(), ResponseRelevancy(), NoiseSensitivity(),
    ]

    from ragas import RunConfig
    run_cfg = RunConfig(**ragas_run_config)

    print(f"evaluate_ragas: avaliando {len(rag_results)} mostras con 6 métricas...")
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_emb, run_config=run_cfg)

    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    print("Scores medios:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    # MLflow non admite paréntese nin '=' nos nomes de métrica; RAGAS úsaos
    # (p.ex. "noise_sensitivity(mode=relevant)"). Saneamos só para o logging.
    def _mlflow_name(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z_./ -]", "_", name).strip("_")

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=params["experiment_id"]):
        mlflow.log_params({
            "chunk_strategy":     params["chunking"]["strategy"],
            "chunk_size":         params["chunking"]["chunk_size"],
            "chunk_overlap":      params["chunking"]["chunk_overlap"],
            "embedding_model":    params["embedding"]["model"],
            "retrieval_strategy": params["retrieval"]["strategy"],
            "retrieval_top_k":    params["retrieval"]["top_k"],
            "llm_model":          params["llm"]["model"],
            "ragas_judge_llm":    llm_cfg["model"],
            "ragas_judge_emb":    ragas_embedding,
        })
        mlflow.log_metrics({_mlflow_name(k): float(v) for k, v in scores.items()
                            if not (isinstance(v, float) and v != v)})

    return scores
