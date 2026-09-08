import json


def select_copyright(corpus_clean: list[str], params: dict) -> list[str]:
    keywords   = params["keywords"]
    forms_keep = set(params["forms_keep"])
    min_words  = params["min_words"]

    out = []
    for line in corpus_clean:
        if not line.strip():
            continue
        doc   = json.loads(line)
        title = doc["title"].lower()
        if not any(k in title for k in keywords):
            continue
        if doc["form"] not in forms_keep:
            continue
        if len(doc["text"].split()) < min_words:
            continue
        out.append(line)

    print(f"corpus_copyright: {len(out):,} docs")
    return out
