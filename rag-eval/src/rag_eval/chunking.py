import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Enrutado estrutural por tipo de documento EUR-Lex (campo `form`)
_PARA_FORMS = {"Judgment", "Opinion of the Advocate General", "Order"}
_ART_FORMS = {"Directive", "Regulation", "Proposal for a directive",
              "Amended proposal for a directive", "Proposal for a regulation", "Consolidated text"}

_SENT = re.compile(r"(?<=[.!?])\s+")


def _meta(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "text"}


def _tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def _recursive(cfg: dict, tokenizer) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_corpus(docs: list[dict], cfg: dict) -> list[dict]:
    """docs: dicts con 'text' + metadata. Devolve [{page_content, metadata}]."""
    strategy = cfg["strategy"]
    if strategy == "fixed_size":
        return _fixed(docs, cfg)
    if strategy == "sentence":
        return _sentence(docs, cfg)
    if strategy == "structural":
        return _structural(docs, cfg)
    if strategy == "semantic":
        return _semantic(docs, cfg)
    raise NotImplementedError(f"Chunking strategy '{strategy}' non implementada")


def _fixed(docs: list[dict], cfg: dict) -> list[dict]:
    splitter = _recursive(cfg, _tokenizer(cfg["tokenizer_model"]))
    out = []
    for doc in docs:
        md = _meta(doc)
        out += [{"page_content": s, "metadata": md} for s in splitter.split_text(doc["text"])]
    return out


def _sentence(docs: list[dict], cfg: dict) -> list[dict]:
    tok = _tokenizer(cfg["tokenizer_model"])
    target = cfg["chunk_size"]
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))
    out = []
    for doc in docs:
        md = _meta(doc)
        sents = [s for s in _SENT.split(doc["text"]) if s.strip()]
        cur, cur_t = [], 0
        for s in sents:
            st = ntok(s)
            if cur and cur_t + st > target:
                out.append({"page_content": " ".join(cur), "metadata": md})
                cur, cur_t = [], 0
            cur.append(s); cur_t += st
        if cur:
            out.append({"page_content": " ".join(cur), "metadata": md})
    return out


def _structural_units(text: str, form: str) -> list[str]:
    if form in _PARA_FORMS:
        u = [p for p in re.split(r"(?m)(?=^\d+\.\s)", text) if p.strip()]
    elif form in _ART_FORMS:
        u = [p for p in re.split(r"(?m)(?=^Article\s+\d+)", text) if p.strip()]
    else:
        u = []
    if len(u) <= 1:  # sen marcadores → bloques de parágrafo
        u = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return u


def _parent(units: list[str], sizes: list[int], i: int, parent_size: int) -> str:
    idxs, total, lo, hi = [i], sizes[i], i - 1, i + 1
    while total < parent_size:
        added = False
        if hi < len(units) and total + sizes[hi] <= parent_size:
            idxs.append(hi); total += sizes[hi]; hi += 1; added = True
        if lo >= 0 and total + sizes[lo] <= parent_size:
            idxs.append(lo); total += sizes[lo]; lo -= 1; added = True
        if not added:
            break
    return "\n\n".join(units[j] for j in sorted(idxs))


def _structural(docs: list[dict], cfg: dict) -> list[dict]:
    tok = _tokenizer(cfg["tokenizer_model"])
    maxtok = cfg["chunk_size"]
    pc = cfg["parent_child"]
    parent_size = cfg["parent_size"] if pc else None
    sub = _recursive(cfg, tok)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))
    out = []
    for doc in docs:
        md = _meta(doc)
        units = _structural_units(doc["text"], doc.get("form", ""))
        sizes = [ntok(u) for u in units]
        for i, u in enumerate(units):
            children = sub.split_text(u) if sizes[i] > maxtok else [u]
            parent_text = _parent(units, sizes, i, parent_size) if pc else None
            for c in children:
                cmd = dict(md)
                if parent_text is not None:
                    cmd["parent_text"] = parent_text
                out.append({"page_content": c, "metadata": cmd})
    return out


def _semantic(docs: list[dict], cfg: dict) -> list[dict]:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings

    # Embedder LOCAL lixeiro só para detectar breakpoints (non dispara API).
    emb = HuggingFaceEmbeddings(model_name=cfg["breakpoint_model"])
    chunker = SemanticChunker(emb, breakpoint_threshold_type=cfg["breakpoint_type"])
    tok = _tokenizer(cfg["tokenizer_model"])
    maxtok = cfg["chunk_size"]
    sub = _recursive(cfg, tok)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))
    out = []
    for doc in docs:
        md = _meta(doc)
        for s in chunker.split_text(doc["text"]):
            pieces = sub.split_text(s) if ntok(s) > maxtok else [s]
            out += [{"page_content": p, "metadata": md} for p in pieces]
    return out
