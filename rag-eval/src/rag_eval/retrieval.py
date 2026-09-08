def _rrf(ranked_lists, k: int, c: int = 60):
    """Reciprocal Rank Fusion sobre listas de Documents (clave: page_content)."""
    scores, store = {}, {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            store[key] = doc
    best = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
    return [store[x] for x in best]


class _BM25Index:
    """BM25 (rank_bm25) sobre todo o corpus de chunks da colección."""

    def __init__(self, vectorstore):
        from rank_bm25 import BM25Okapi
        client, coll = vectorstore.client, vectorstore.collection_name
        self.docs, offset = [], None
        while True:
            points, offset = client.scroll(coll, limit=1000, offset=offset, with_payload=True)
            for p in points:
                self.docs.append((p.payload.get("page_content", ""), p.payload.get("metadata", {})))
            if offset is None:
                break
        print(f"  BM25 construído sobre {len(self.docs)} chunks")
        self.bm25 = BM25Okapi([d[0].lower().split() for d in self.docs])

    def top(self, query: str, n: int):
        import numpy as np
        from langchain_core.documents import Document
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:n]
        return [Document(page_content=self.docs[i][0], metadata=self.docs[i][1]) for i in idx]


class _BM25Retriever:
    def __init__(self, vectorstore, top_k: int):
        self.idx = _BM25Index(vectorstore)
        self.k = top_k

    def invoke(self, query: str):
        return self.idx.top(query, self.k)


class _HybridRetriever:
    """BM25 + denso, fusionados con RRF. Sen clases de langchain (env con core 1.x)."""

    def __init__(self, vectorstore, top_k: int):
        self.vs = vectorstore
        self.k = top_k
        self.fetch = max(top_k * 4, 20)
        self.idx = _BM25Index(vectorstore)

    def invoke(self, query: str):
        dense = self.vs.similarity_search(query, k=self.fetch)
        sparse = self.idx.top(query, self.fetch)
        return _rrf([dense, sparse], self.k)


class _CrossEncoderRerank:
    """2ª etapa: cross-encoder local (bge-reranker-v2-m3) repuntúa pares (query, doc)."""

    def __init__(self, base, model: str, top_k: int):
        from sentence_transformers import CrossEncoder
        self.base = base
        self.ce = CrossEncoder(model)
        self.k = top_k

    def invoke(self, query: str):
        docs = self.base.invoke(query)
        if not docs:
            return docs
        scores = self.ce.predict([(query, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[:self.k]]


class _LLMRerank:
    """2ª etapa: LLM (DeepSeek) listwise — 1 chamada por query, devolve a orde."""

    def __init__(self, base, llm_cfg: dict, top_k: int):
        self.base = base
        self.cfg = llm_cfg
        self.k = top_k

    def invoke(self, query: str):
        import os, re
        from langchain_openai import ChatOpenAI
        docs = self.base.invoke(query)
        if not docs:
            return docs
        llm = ChatOpenAI(model=self.cfg["model"], temperature=0.0,
                         base_url=self.cfg["base_url"], api_key=os.environ["DEEPSEEK_API_KEY"],
                         timeout=60, max_retries=1)
        listing = "\n".join(f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs))
        prompt = (f"Question: {query}\n\nPassages:\n{listing}\n\n"
                  f"Rank the passages by relevance to the question. Return ONLY a JSON array with the "
                  f"{self.k} most relevant passage indices, best first. Example: [3, 0, 7]")
        resp = llm.invoke(prompt).content
        idx, seen = [], set()
        for x in re.findall(r"\d+", resp):
            i = int(x)
            if 0 <= i < len(docs) and i not in seen:
                idx.append(i); seen.add(i)
        # robustez de parseo: se o LLM devolve algo inservible, mantemos a orde base
        for i in range(len(docs)):
            if i not in seen:
                idx.append(i); seen.add(i)
        return [docs[i] for i in idx[:self.k]]


def make_reranker(base, cfg: dict):
    if cfg["type"] == "cross_encoder":
        return _CrossEncoderRerank(base, cfg["model"], cfg["top_k"])
    if cfg["type"] == "llm":
        return _LLMRerank(base, cfg["llm"], cfg["top_k"])
    raise NotImplementedError(f"Reranker type '{cfg['type']}' non implementado")


def get_retriever(vectorstore, strategy: str, params: dict):
    top_k = params["top_k"]

    if strategy == "cosine":
        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    if strategy == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": params["fetch_k"]},
        )

    if strategy == "bm25":
        return _BM25Retriever(vectorstore, top_k)

    if strategy == "hybrid":
        return _HybridRetriever(vectorstore, top_k)

    if strategy in ("multi_query", "hyde"):
        raise NotImplementedError(f"'{strategy}' é query-expansion (exp4), non unha estratexia de retrieval")
    raise NotImplementedError(f"Retrieval strategy '{strategy}' non implementada")


def candidate_retriever(vectorstore, rcfg: dict, fetch_k: int, pool: int):
    """1ª/2ª etapa da cascada de rerank: entrega `fetch_k` candidatos co
    retriever gañador. Só mmr usa `pool` (recupera pool por similitude e
    selecciona fetch_k diversos); o resto entrega top-fetch_k directo."""
    s = rcfg["strategy"]
    if s == "cosine":
        return vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    if s == "bm25":
        return _BM25Retriever(vectorstore, fetch_k)
    if s == "hybrid":
        return _HybridRetriever(vectorstore, fetch_k)
    if s == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": fetch_k, "fetch_k": pool})
    raise NotImplementedError(f"Estratexia '{s}' non soportada como base de rerank")
