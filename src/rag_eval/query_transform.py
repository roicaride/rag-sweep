"""Query transformation / expansion (pre-retrieval) — etapa upstream do pipeline.

Equivalente manual aos retrievers de LangChain (MultiQuery, HyDE, SelfQuery…),
inalcanzables neste env (langchain.retrievers choca con core 1.x). Cada técnica
transforma a query cun LLM (DeepSeek) e recupera co retriever base (mmr
gañador), reutilizando `get_retriever` e `_rrf` de retrieval.py."""

from rag_eval.retrieval import get_retriever, _rrf


_PROMPTS = {
    "rewrite": (
        "Rewrite the following question to be clearer and more effective for "
        "retrieving relevant documents from a legal corpus on EU copyright law. "
        "Keep the same intent. Return ONLY the rewritten question.\n\nQuestion: {q}"),
    "hyde": (
        "Write a short hypothetical passage (3-5 sentences) from an EU legal "
        "document that would directly answer the following question. Write it as if "
        "it were an excerpt from the actual document. Return ONLY the passage.\n\n"
        "Question: {q}"),
    "stepback": (
        "Given the following specific question, generate a more general 'step-back' "
        "question that captures the broader concept or principle needed to answer "
        "it. Return ONLY the step-back question.\n\nQuestion: {q}"),
    "multiquery": (
        "Generate {n} different versions of the following question to retrieve "
        "relevant documents from a vector database on EU copyright law. Use diverse "
        "phrasings and perspectives. Return ONLY the {n} questions, one per line, "
        "numbered.\n\nQuestion: {q}"),
    "decomposition": (
        "Break the following question into {n} simpler sub-questions that together "
        "cover what is needed to answer it. Return ONLY the sub-questions, one per "
        "line, numbered.\n\nQuestion: {q}"),
    "selfquery": (
        "You convert a user question into a semantic search query plus optional "
        "metadata filters for a corpus of EU copyright legal documents.\n\n"
        "Filter fields (use an EXACT value from the list, or null):\n"
        "- family: one of [JUDICIAL, POLICY, LEGISLATIVE, OTHER]\n"
        "- form: one of [Judgment, Directive, Opinion of the Advocate General, "
        "Written question, Impact assessment, Proposal for a directive, "
        "Communication, Green Paper, Opinion, Decision, Order, Consolidated text]\n\n"
        "Set a filter ONLY when the question clearly implies it (e.g. 'in case law' "
        "-> family JUDICIAL; 'which directive' -> form Directive). Otherwise null.\n"
        "Return ONLY a JSON object: "
        '{{"query": "<clean semantic query>", "family": <value or null>, '
        '"form": <value or null>}}\n\nQuestion: {q}'),
}
_PROMPTS["rag_fusion"] = _PROMPTS["multiquery"]  # mesmas variantes; difiren só na fusión (RRF)


def _qt_llm(cfg: dict):
    import os
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=cfg["model"], temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
        base_url=cfg["base_url"], api_key=os.environ["DEEPSEEK_API_KEY"],
        timeout=60, max_retries=1)


def _parse_lines(text: str, n: int) -> list:
    """Extrae ata n entradas dunha lista numerada/con guións do output do LLM."""
    import re
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", ln.strip()).strip()
        if ln:
            out.append(ln)
    return out[:n]


def _interleave_dedup(lists, k: int) -> list:
    """Unión round-robin de varias listas de Documents, dedup por contido, top-k."""
    seen, out = set(), []
    depth = max((len(l) for l in lists), default=0)
    for rank in range(depth):
        for l in lists:
            if rank < len(l):
                key = l[rank].page_content
                if key not in seen:
                    seen.add(key)
                    out.append(l[rank])
                    if len(out) >= k:
                        return out
    return out


class _QueryTransformRetriever:
    """Transforma a query cun LLM e recupera co retriever base (rcfg)."""

    def __init__(self, vectorstore, rcfg: dict, cfg: dict, out_k: int = None):
        self.vs = vectorstore
        self.rcfg = rcfg
        self.cfg = cfg
        self.type = cfg["type"]
        # out_k: tamaño de saída. None → top_k do retrieval; en modo rerank pásase
        # rr["fetch_k"] para que a transform entregue o POOL que reordena o reranker.
        self.top_k = out_k or rcfg["top_k"]
        self.llm = _qt_llm(cfg["llm"])
        self.base = get_retriever(vectorstore, rcfg["strategy"], {**rcfg, "top_k": self.top_k})

    def _gen(self, query: str) -> str:
        return self.llm.invoke(_PROMPTS[self.type].format(q=query, n=self.cfg.get("n_queries"))).content

    def invoke(self, query: str):
        t = self.type
        if t == "rewrite":
            return self.base.invoke(self._gen(query).strip())
        if t == "hyde":
            return self.base.invoke(self._gen(query).strip())
        if t == "stepback":
            sb = self._gen(query).strip()
            return _rrf([self.base.invoke(query), self.base.invoke(sb)], self.top_k)
        if t in ("multiquery", "rag_fusion"):
            variants = _parse_lines(self._gen(query), self.cfg["n_queries"])
            lists = [self.base.invoke(q) for q in [query] + variants]
            return _rrf(lists, self.top_k) if t == "rag_fusion" else _interleave_dedup(lists, self.top_k)
        if t == "decomposition":
            subs = _parse_lines(self._gen(query), self.cfg["n_queries"])
            return _rrf([self.base.invoke(s) for s in subs or [query]], self.top_k)
        if t == "selfquery":
            return self._selfquery(query)
        raise NotImplementedError(f"query_transform type '{t}' non implementado")

    def _selfquery(self, query: str):
        import json, re
        from qdrant_client import models
        m = re.search(r"\{.*\}", self._gen(query), re.S)
        spec = json.loads(m.group(0)) if m else {}
        conds = [
            models.FieldCondition(key=f"metadata.{f}", match=models.MatchValue(value=spec[f]))
            for f in ("family", "form") if spec.get(f)
        ]
        qfilter = models.Filter(must=conds) if conds else None
        # mmr + filtro: mesma base de recuperación que o resto de técnicas (non densa pura)
        return self.vs.max_marginal_relevance_search(
            spec.get("query") or query, k=self.top_k, fetch_k=self.rcfg["fetch_k"], filter=qfilter)


def query_transform_retriever(vectorstore, rcfg: dict, cfg: dict, out_k: int = None):
    return _QueryTransformRetriever(vectorstore, rcfg, cfg, out_k)
