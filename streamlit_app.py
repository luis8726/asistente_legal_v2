import os
import time
import json
import requests
import streamlit as st

from legal_core import retrieve, build_context, generate_answer, rows_to_sources

st.set_page_config(
    page_title="Asistente Legal",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Asistente Legal (Vector Search + OpenAI)")

# =====================================================
# CONFIGURACIÓN FIJA (antes venía de la sidebar)
# =====================================================
show_sources = True          # ← antes checkbox
show_context = False         # ← antes checkbox

# =====================================================
# (DIAGNÓSTICOS – DESHABILITADOS / COMENTADOS)
# =====================================================

def diag_env():
    keys = ["OPENAI_API_KEY", "DATABRICKS_TOKEN", "LLM_MODEL", "PYTHON_VERSION"]
    out = {}
    for k in keys:
        v = os.environ.get(k)
        out[k] = "OK" if v else "MISSING"
    return out

def diag_databricks():
    host = os.environ.get("DATABRICKS_HOST", "https://dbc-999eea35-2964.cloud.databricks.com").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    index_name = os.environ.get("INDEX_FULL_NAME", "chalk_workspace.legales.kb_laws_chunks_vs_index_v3")

    url = f"{host}/api/2.0/vector-search/indexes/{index_name}"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers, timeout=30)
    return {"status": r.status_code, "body": r.text[:1200], "url": url}

def diag_openai():
    from openai import OpenAI
    import httpx

    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gpt-5-mini")

    client = OpenAI(api_key=key, timeout=httpx.Timeout(60.0, connect=20.0))
    t0 = time.time()
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": "decí OK"}],
    )
    return {
        "ok": True,
        "model": model,
        "seconds": round(time.time() - t0, 2),
        "text": resp.output_text[:200],
    }

# =====================================================
# CHAT
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Escribí tu consulta legal…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base legal…"):
            try:
                res = retrieve(prompt)
                context = build_context(res)

                if not context.strip():
                    answer = "No encontré evidencia suficiente en los documentos recuperados."
                    sources = []
                else:
                    answer = generate_answer(prompt, context)
                    sources = rows_to_sources(res)

                st.markdown(answer)

                if show_sources and sources:
                    st.markdown("**Fuentes (recuperadas):**")
                    for s in sources:
                        st.write(
                            f"- doc_id={s['doc_id']} | "
                            f"art={s['article_number']} | "
                            f"law={s['law_number']} | "
                            f"subchunk={s['subchunk_id']}"
                        )

                if show_context and context:
                    with st.expander("🔎 Contexto recuperado (debug)"):
                        st.text(context)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error("Fallo en ejecución (detalle abajo).")
                st.exception(e)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"ERROR: {repr(e)}"}
                )

