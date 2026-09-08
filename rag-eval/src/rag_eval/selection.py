"""Selección automática do gañador de cada fase.

Criterio único e consistente para todas as fases: **recall + precision**
(as dúas métricas de retrieval, pesadas 1:1). Aplícase igual a embeddings
(exp1) e a chunking (exp2)."""


def _score(m: dict) -> float:
    return m["context_recall"] + m["context_precision"]


def _rank(cands: list) -> list:
    """cands: [(label, metrics, payload)] → ordenados por recall+precision desc."""
    return sorted(cands, key=lambda c: _score(c[1]), reverse=True)


def _report(title: str, ranked: list):
    print(f"{title} (criterio: recall+precision):")
    for label, m, _ in ranked:
        print(f"  {label:42s} score={_score(m):.4f}  (recall={m['context_recall']:.4f} precision={m['context_precision']:.4f})")
    print(f"  gañador: {ranked[0][0]}")


def select_best_embedding(m_exp0, m_snow, m_bge_m3, m_qwen,
                          p_exp0, p_snow, p_bge_m3, p_qwen) -> str:
    """Devolve o nome do modelo de embedding gañador de exp1."""
    cands = [
        (p_exp0["embedding"]["model"], m_exp0, p_exp0),
        (p_snow["embedding"]["model"], m_snow, p_snow),
        (p_bge_m3["embedding"]["model"], m_bge_m3, p_bge_m3),
        (p_qwen["embedding"]["model"], m_qwen, p_qwen),
    ]
    ranked = _rank(cands)
    _report("select_best_embedding", ranked)
    return ranked[0][0]


def select_best_chunking(m_256, m_512, m_sent, m_struct, m_pc, m_sem,
                         p_256, p_512, p_sent, p_struct, p_pc, p_sem) -> dict:
    """Devolve a CONFIG de chunking gañadora de exp2 (incl. 256 = exp1__bge_m3)."""
    cands = [
        (p_256["experiment_id"], m_256, p_256),
        (p_512["experiment_id"], m_512, p_512),
        (p_sent["experiment_id"], m_sent, p_sent),
        (p_struct["experiment_id"], m_struct, p_struct),
        (p_pc["experiment_id"], m_pc, p_pc),
        (p_sem["experiment_id"], m_sem, p_sem),
    ]
    ranked = _rank(cands)
    _report("select_best_chunking", ranked)
    return ranked[0][2]["chunking"]


def select_best_retrieval(*args) -> dict:
    """Elixe o RETRIEVER (estratexia) robusto a k: agrupa por estratexia,
    promedia recall+precision sobre as variantes k in {5, 10} (mesmos puntos
    para todas), e devolve a config da estratexia gañadora. O k final NON
    se decide aquí — fíxase no reranking. Inputs: N eval_metrics + N params."""
    n = len(args) // 2
    metrics, params = args[:n], args[n:]
    groups, rep = {}, {}
    for m, p in zip(metrics, params):
        r = p["retrieval"]
        if r.get("top_k") not in (5, 10):
            continue
        groups.setdefault(r["strategy"], []).append(_score(m))
        if r["strategy"] not in rep or r["top_k"] == 10:  # representativo: preferimos k=10
            rep[r["strategy"]] = r
    means = {s: sum(v) / len(v) for s, v in groups.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    print("select_best_retrieval (media recall+precision sobre k=5,10):")
    for s in ranked:
        print(f"  {s:10s} media={means[s]:.4f}  (n={len(groups[s])})")
    print(f"  gañador: {ranked[0]}")
    return rep[ranked[0]]


def select_best_rerank(*args) -> dict:
    """Elixe o reranking gañador de exp5 por recall+precision. Inclúe o
    baseline SEN rerank (mmr_k10) como candidato: só se elixe rerank se mellora.
    Devolve a config 'rerank' do gañador, ou {} se gana o non-rerank.
    Inputs: N eval_metrics + N params (mesma orde)."""
    n = len(args) // 2
    metrics, params = args[:n], args[n:]
    cands = [(p["experiment_id"], m, p) for m, p in zip(metrics, params)]
    ranked = _rank(cands)
    _report("select_best_rerank", ranked)
    return ranked[0][2].get("rerank", {})


def select_best_query_transform(*args) -> dict:
    """Elixe a técnica de query-expansion gañadora de exp4a–4g por recall+precision.
    Inclúe como referencia o baseline mmr_k10 (sen transform) e o campión actual
    (rerank ce_k10): só se elixe unha técnica se supera a ambos. Devolve a config
    'query_transform' do gañador, ou {} se gana unha referencia (non-transform).
    Inputs: N eval_metrics + N params (mesma orde)."""
    n = len(args) // 2
    metrics, params = args[:n], args[n:]
    cands = [(p["experiment_id"], m, p) for m, p in zip(metrics, params)]
    ranked = _rank(cands)
    _report("select_best_query_transform", ranked)
    return ranked[0][2].get("query_transform", {})
