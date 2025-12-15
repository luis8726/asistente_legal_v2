import os, re, json, requests
from urllib.parse import quote
from openai import OpenAI

DATABRICKS_HOST = "https://dbc-999eea35-2964.cloud.databricks.com"
INDEX_FULL_NAME = "chalk_workspace.legales.kb_laws_chunks_vs_index_v3"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno")

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "").strip()
if not DATABRICKS_TOKEN:
    raise RuntimeError("Falta DATABRICKS_TOKEN en variables de entorno")

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.2")

# columnas reales (NO incluyas score)
COLUMNS = ["chunk_text", "doc_id", "article_number", "law_number", "subchunk_id"]

TOP_K_GENERAL = 20
TOP_K_ARTICLE = 30

oai = OpenAI(api_key=OPENAI_API_KEY)

DOCID_RULES = [
    (r"\b27\.?349\b|\b27349\b", "ley_27349"),
    (r"\b19\.?550\b|\b19550\b|\blgs\b", "lgs_19550"),
    (r"\bccycn\b|\bc[oó]digo civil\b|\bc[oó]digo civil y comercial\b", "ccycn_1251_fin"),
]

SYSTEM_PROMPT = """Eres un asistente legal argentino con grounding estricto.

Reglas:
1) Responde SOLO usando el CONTEXTO recuperado.
2) No uses conocimiento externo. No inventes artículos, números ni requisitos.
3) Si el contexto NO alcanza para responder, di exactamente: "No encontré evidencia suficiente en los documentos recuperados."
4) Siempre agrega una sección "Fuentes" citando doc_id y article_number utilizados.
5) Si el contexto habla de derogaciones/vigencia, menciona eso.

Formato:
- Respuesta
- Fundamento
- Fuentes
"""

def parse_intent(q: str) -> dict:
    ql = q.lower()
    m = re.search(r"\b(art\.?|artículo)\s*(\d+)\b", ql)
    article = int(m.group(2)) if m else None

    doc_id = None
    for pat, did in DOCID_RULES:
        if re.search(pat, ql):
            doc_id = did
            break
    return {"article_number": article, "doc_id": doc_id}

def embed(text: str) -> list[float]:
    r = oai.embeddings.create(model=EMBEDDING_MODEL, input=text.strip())
    vec = r.data[0].embedding
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(f"Embedding dim {len(vec)} != {EMBEDDING_DIM}")
    return vec

def vs_query(payload: dict) -> dict:
    idx = quote(INDEX_FULL_NAME, safe="")
    url = f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{idx}/query"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if not r.ok:
        raise RuntimeError(f"VS ERROR {r.status_code}: {r.text[:1500]}")
    return r.json()

def retrieve(question: str) -> dict:
    intent = parse_intent(question)

    # Caso artículo: FULL_TEXT + fallback exacto
    if intent["article_number"] is not None and intent["doc_id"] is not None:
        payload_ft = {
            "query_type": "FULL_TEXT",
            "query_text": f"Artículo {int(intent['article_number'])}",
            "num_results": TOP_K_ARTICLE,
            "columns": COLUMNS,
            "filters": json.dumps({"doc_id": intent["doc_id"]})
        }
        res = vs_query(payload_ft)
        if res["result"]["row_count"] > 0:
            payload_exact = {
                "query_vector": embed(question),
                "query_type": "ANN",
                "num_results": TOP_K_ARTICLE,
                "columns": COLUMNS,
                "filters": json.dumps({
                    "doc_id": intent["doc_id"],
                    "article_number": float(intent["article_number"])
                })
            }
            res_exact = vs_query(payload_exact)
            return res_exact if res_exact["result"]["row_count"] > 0 else res

        payload_exact = {
            "query_vector": embed(question),
            "query_type": "ANN",
            "num_results": TOP_K_ARTICLE,
            "columns": COLUMNS,
            "filters": json.dumps({
                "doc_id": intent["doc_id"],
                "article_number": float(intent["article_number"])
            })
        }
        res = vs_query(payload_exact)
        if res["result"]["row_count"] > 0:
            return res

        payload_doc = {
            "query_vector": embed(question),
            "query_type": "ANN",
            "num_results": 40,
            "columns": COLUMNS,
            "filters": json.dumps({"doc_id": intent["doc_id"]})
        }
        return vs_query(payload_doc)

    # Caso general
    payload = {
        "query_vector": embed(question),
        "query_type": "ANN",
        "num_results": TOP_K_GENERAL,
        "columns": COLUMNS
    }
    if intent["doc_id"] is not None:
        payload["filters"] = json.dumps({"doc_id": intent["doc_id"]})
        payload["num_results"] = max(TOP_K_GENERAL, 30)

    res = vs_query(payload)
    if res["result"]["row_count"] == 0 and "filters" in payload:
        payload.pop("filters", None)
        payload["num_results"] = 40
        res = vs_query(payload)
    return res

def rows_to_sources(res: dict, max_items: int = 8) -> list[dict]:
    cols = [c["name"] for c in res["manifest"]["columns"]]
    out = []
    for row in res["result"]["data_array"][:max_items]:
        d = dict(zip(cols, row))
        out.append({
            "doc_id": d.get("doc_id"),
            "article_number": d.get("article_number"),
            "law_number": d.get("law_number"),
            "subchunk_id": d.get("subchunk_id"),
        })
    return out

def build_context(res: dict, max_chars: int = 14000) -> str:
    cols = [c["name"] for c in res["manifest"]["columns"]]
    rows = res["result"]["data_array"]
    parts, total = [], 0

    for row in rows:
        d = dict(zip(cols, row))
        txt = (d.get("chunk_text") or "").strip()
        if not txt:
            continue
        meta = f"[doc_id={d.get('doc_id')} | art={d.get('article_number')} | subchunk={d.get('subchunk_id')}]"
        block = meta + "\n" + txt
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)

def generate_answer(question: str, context: str) -> str:
    user = f"""PREGUNTA:
{question}

CONTEXTO (fragmentos recuperados):
{context}
"""
    resp = oai.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return resp.output_text




