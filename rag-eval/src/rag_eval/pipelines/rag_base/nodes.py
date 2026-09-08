import copy
import json
import time

from langchain_core.documents import Document

from rag_eval.chunking import chunk_corpus
from rag_eval.embedding import get_embeddings


def resolve_params(params: dict, best_embedding: str, best_chunking: dict = None,
                   best_retrieval: dict = None) -> dict:
    """Enche as dimensións arrastradas (embedding, chunking, retrieval) co
    gañador da fase anterior se non están fixadas explicitamente en parameters.
    Override por params; default = gañador. Cada best_* é opcional: pásase dende
    a fase que xa decidiu esa dimensión en diante."""
    p = copy.deepcopy(params)
    if not p.get("embedding", {}).get("model"):
        p.setdefault("embedding", {})["model"] = best_embedding
    if best_chunking is not None and not p.get("chunking"):
        p["chunking"] = best_chunking
    if best_retrieval is not None and not p.get("retrieval"):
        p["retrieval"] = best_retrieval
    chunk_desc = p["chunking"]["strategy"] if "chunking" in p else "(n/a)"
    retr_desc = f"{p['retrieval']['strategy']} k={p['retrieval'].get('top_k')}" if "retrieval" in p else "(n/a)"
    print(f"resolve_params [{p.get('experiment_id')}]: embedding={p['embedding']['model']}, chunking={chunk_desc}, retrieval={retr_desc}")
    return p


def chunk_documents(corpus: list[str], params: dict) -> list[dict]:
    docs   = [json.loads(l) for l in corpus if l.strip()]
    chunks = chunk_corpus(docs, params["chunking"])
    print(f"chunk_documents [{params['chunking']['strategy']}]: {len(docs)} docs → {len(chunks)} chunks")
    return chunks


def build_index(chunks: list[dict], params: dict, qdrant_url: str):
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    model_name    = params["embedding"]["model"]
    batch_size    = params["embedding"]["batch_size"]
    experiment_id = params["experiment_id"]

    embeddings  = get_embeddings(model_name)
    client      = QdrantClient(url=qdrant_url)
    collections = [c.name for c in client.get_collections().collections]

    # Colección completa xa existente → reutilizar. Se quedou parcial dun fallo previo, recrear.
    if experiment_id in collections:
        existing = client.count(collection_name=experiment_id).count
        if existing == len(chunks):
            print(f"build_index: colección '{experiment_id}' completa ({existing} chunks), reutilizo.")
            return experiment_id
        print(f"build_index: colección '{experiment_id}' parcial ({existing}/{len(chunks)}), recréoa.")
        client.delete_collection(collection_name=experiment_id)

    dim = len(embeddings.embed_query("dimension check"))
    client.create_collection(
        collection_name=experiment_id,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    lc_docs = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks]
    total   = len(lc_docs)
    print(f"build_index: indexando {total} chunks ({model_name}, dim={dim}, batch={batch_size})...")

    vs = QdrantVectorStore(client=client, collection_name=experiment_id, embedding=embeddings)
    for i in range(0, total, batch_size):
        batch   = lc_docs[i:i + batch_size]
        retries = 0
        while True:
            try:
                vs.add_documents(batch)
                break
            except Exception as e:
                msg = str(e).lower()
                is_rate  = "429" in msg or "resource_exhausted" in msg or "quota" in msg
                is_trans = any(s in msg for s in ("500", "502", "503", "504", "gateway", "time-out", "timeout", "temporarily"))
                max_tries = 6 if is_rate else 15  # 504/timeout son frecuentes pero recuperables → máis paciencia
                if (is_rate or is_trans) and retries < max_tries:
                    wait = 65 if is_rate else 12
                    retries += 1
                    print(f"  Erro transitorio (agardo {wait}s, intento {retries}/{max_tries}): {str(e)[:90]}")
                    time.sleep(wait)
                else:
                    raise
        done = min(i + batch_size, total)
        if done % 500 == 0 or done == total:
            print(f"  {done}/{total} chunks indexados")
        if done < total:
            time.sleep(0.8)

    print("build_index: índice construído e gardado en disco.")
    return experiment_id
