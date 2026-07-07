import json
import re


def clean_text(content: str) -> str:
    text = _strip_header(content)
    text = _remove_footnote_anchors(text)
    text = _remove_top_link(text)
    text = _strip_links(text)
    text = _strip_markup(text)
    text = _flatten_two_col_tables(text)
    text = _normalize_whitespace(text)
    return text.strip()


def _strip_header(text: str) -> str:
    # Saltamos as liñas de cabeceira/navegación (táboas, separadores, ligazóns, markup).
    # Devolvemos a partir da primeira liña que pareza contido real do documento.
    lines = text.split('\n')
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not s.startswith(('|', '#', '[', '*', '-', '!')) and len(s) > 15:
            return '\n'.join(lines[i:])
    return text


def _remove_footnote_anchors(text: str) -> str:
    # [(1)](#ntr1-...) e [(*1)](#ntr*1-...) — áncoras HTML internas
    return re.sub(r'\[\([*\d]+\)\]\(#nt[rc][^)]*\)', '', text)


def _remove_top_link(text: str) -> str:
    # Eliminamos a ligazón de navegación e o --- orfo que a precedía
    text = re.sub(r'\[Top\]\(#[^)]*\)', '', text)
    text = re.sub(r'(\s*---\s*)+$', '', text)
    return text


def _strip_links(text: str) -> str:
    # ![alt](url) → '' (as imaxes non achegan valor textual nos documentos legais)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # [texto visible](url) → texto visible
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    return text


def _strip_markup(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)            # **bold** → bold
    text = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', text) # *italic* → italic
    text = re.sub(r'`([^`]+)`', r'\1', text)                   # `code` → code
    text = re.sub(r'(?m)^#{1,6}[ \t]+', '', text)              # # Heading → Heading
    return text


def _flatten_two_col_tables(text: str) -> str:
    # Eliminamos a fila cabeceira baleira de 2 columnas:  |  |  |
    text = re.sub(r'(?m)^\|[ \t]*\|[ \t]*\|[ \t]*$', '', text)
    # Eliminamos a fila separadora de 2 columnas:  | --- | --- |
    text = re.sub(r'(?m)^\|[ \t]*-+[ \t]*\|[ \t]*-+[ \t]*\|[ \t]*$', '', text)
    # Aplanamos calquera fila de contido de 2 columnas restante:  | col1 | col2 |  →  col1 col2
    text = re.sub(
        r'(?m)^\|[ \t]*([^|\n]+?)[ \t]*\|[ \t]*([^|\n]+?)[ \t]*\|[ \t]*$',
        lambda m: f'{m.group(1).strip()} {m.group(2).strip()}',
        text,
    )
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = re.sub(r'(?m)^[ \t]+$', '', text)   # liñas só con espazos en branco → baleiras
    text = re.sub(r'(?m)(^---\s*$\n*){2,}', '---\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

# ── Método 1: sector CELEX (primeiro carácter) → family ───────────────────────
# Sectores con asignación directa
_SECTOR_FAMILY = {
    '0': 'LEGISLATIVE', '1': 'LEGISLATIVE', '2': 'LEGISLATIVE', '4': 'LEGISLATIVE',
    '6': 'JUDICIAL',    '8': 'JUDICIAL',
    '9': 'POLICY',
}
# Sector 3 — type_code → POLICY; resto → LEGISLATIVE
_S3_POLICY_CODES = {'H', 'C', 'XC', 'SC', 'K'}
# Sector 5 — type_code → POLICY; se non, cae no método 2
_S5_POLICY_CODES = {'SC', 'XC', 'AE', 'IP', 'IE'}

# ── Método 2: form do documento → family (só sector 5 sen type_code propio) ────
_POLICY_FORMS = {
    'Communication',                  'Recommendation',
    'Opinion',                        'Staff working document',
    'Impact assessment',              'Report',
    'Resolution',                     'Information',
    'Own-initiative resolution',      'Own-initiative opinion',
    'Green Paper',                    'Evaluation',
    'Written question',               'Legislative resolution',
    'Opinion not proposing amendment','Opinion proposing amendment',
    'Council conclusions',
}


def celex_to_family(celex: str, form: str = '') -> str:
    if not celex:
        return 'OTHER'
    sector = celex[0]
    if sector in _SECTOR_FAMILY:                          # método 1 — directo
        return _SECTOR_FAMILY[sector]
    type_code = re.match(r'[A-Z]+', celex[5:]).group(0) if len(celex) > 5 else ''
    if sector == '3':                                     # método 1 — type_code
        return 'POLICY' if type_code in _S3_POLICY_CODES else 'LEGISLATIVE'
    if sector == '5':                                     # método 1 + método 2
        return 'POLICY' if (type_code in _S5_POLICY_CODES or form in _POLICY_FORMS) else 'OTHER'
    return 'OTHER'


def clean_corpus(raw_corpus: list[str]) -> list[str]:
    """Recibe unha lista de liñas JSONL, devolve unha lista de liñas limpas.

    doc_id: índice secuencial no corpus procesado (0, 1, 2, ...), sen ocos.
    celex:  identificador oficial EUR-Lex, estable entre execucións.
    """
    lines_out = []
    doc_id = 0
    for line in raw_corpus:
        if not line.strip():
            continue
        doc = json.loads(line)
        meta = doc['metadata']
        celex = meta.get('celex', '')

        date_raw = meta.get('date', '')
        if ';' in date_raw:
            date_val, date_type = date_raw.split(';', 1)
        else:
            date_val, date_type = date_raw, ''

        record = {
            'doc_id':    doc_id,
            'celex':     celex,
            'title':     meta.get('title', '').replace('\xa0', ' '),
            'form':      meta.get('form', ''),
            'family':    celex_to_family(celex, meta.get('form', '')),
            'author':    meta.get('author', ''),
            'date':      date_val.strip(),
            'date_type': date_type.strip(),
            'latest':    meta.get('latest', ''),
            'pages':     meta.get('pages', ''),
            'url':       meta.get('source', ''),
            'text':      clean_text(doc['content']),
        }
        lines_out.append(json.dumps(record, ensure_ascii=False))
        doc_id += 1

        if doc_id % 5000 == 0:
            print(f"  {doc_id:,} procesados...")

    print(f"\nTotal: {len(lines_out):,} documentos")
    return lines_out
