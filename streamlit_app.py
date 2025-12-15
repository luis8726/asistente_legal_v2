import os
import time
import json
import requests
import streamlit as st

from legal_core import retrieve, build_context, generate_answer, rows_to_sources

st.set_page_config(page_title="Asistente Legal", page_icon="⚖️", layout="centered")
st.title("⚖️ Asistente Legal (Vector Search + OpenAI)")
"""
# =========================
# DIAGNÓSTICO
# =========================
def diag_env():
    keys = ["OPENAI_API_KEY", "DATABRICKS_TOKEN", "LLM_MODEL", "PYTHON_VERSION"]
    out = {}
    for k in keys:
        v = os.environ.get(k)
        out[k] = "OK" if v else "MISSING"
    return out

def diag_databricks():
    # Prueba simple: listar el índice (GET). Si esto falla, VS no va a andar.
    host = os.environ.get("DATABRICKS_HOST", "https://dbc-999eea35-2964.cloud.databricks.com").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    index_name = os.environ.get("INDEX_FULL_NAME", "chalk_workspace.legales.kb_laws_chunks_vs_index_v3")

    url = f"{host}/api/2.0/vector-search/indexes/{index_name}"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers, timeout=30)
    return {"status": r.status_code, "body": r.text[:1200], "url": url}

def diag_openai():
    # Prueba simple: llamar al endpoint Responses con un input mínimo
    from openai import OpenAI
    import httpx

    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gpt-5-mini")

    client = OpenAI(api_key=key, timeout=httpx.Timeout(60.0, connect=20.0))
    t0 = time.time()
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": "decí OK"}],
        )
        dt = time.time() - t0
        return {"ok": True, "model": model, "seconds": round(dt, 2), "text": resp.output_text[:200]}
    except Exception as e:
        return {"ok": False, "model": model, "error": repr(e)}

with st.sidebar:
    st.header("Opciones")
    show_sources = st.checkbox("Mostrar fuentes", value=True)
    show_context = st.checkbox("Mostrar contexto (debug)", value=False)
    st.divider()
    st.subheader("Diagnóstico")
    if st.button("Correr diagnóstico"):
        st.write("ENV:", diag_env())
        st.write("DATABRICKS:", diag_databricks())
        st.write("OPENAI:", diag_openai())
        st.info("Copiá estos resultados y pegámelos si sigue fallando.")

import socket, requests, os

def net_diag_openai():
    out = {}

    # 1. DNS
    try:
        out["dns_api_openai_com"] = socket.getaddrinfo("api.openai.com", 443)[0][4][0]
    except Exception as e:
        out["dns_api_openai_com"] = f"DNS_FAIL: {repr(e)}"

    # 2. HTTPS sin auth
    try:
        r = requests.get("https://api.openai.com/v1/models", timeout=20)
        out["https_noauth_status"] = r.status_code
        out["https_noauth_body"] = r.text[:120]
    except Exception as e:
        out["https_noauth_status"] = f"HTTPS_FAIL: {repr(e)}"

    # 3. HTTPS con auth
    try:
        #key = os.environ.get("OPENAI_API_KEY", "")
        key = (os.environ.get("OPENAI_API_KEY", "")).strip()
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        out["https_auth_status"] = r.status_code
        out["https_auth_body"] = r.text[:120]
    except Exception as e:
        out["https_auth_status"] = f"HTTPS_AUTH_FAIL: {repr(e)}"

    return out

with st.sidebar:
    st.divider()
    st.subheader("Diagnóstico de red OpenAI")

    if st.button("Probar conexión OpenAI"):
        result = net_diag_openai()
        st.write(result)
        st.info("Copiá este resultado y pegámelo en el chat.")
"""
# =========================
# CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

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
                        st.write(f"- doc_id={s['doc_id']} | art={s['article_number']} | law={s['law_number']} | subchunk={s['subchunk_id']}")

                if show_context and context:
                    with st.expander("🔎 Contexto recuperado (debug)"):
                        st.text(context)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                # MOSTRAR EL ERROR REAL
                st.error("Fallo en ejecución (detalle abajo).")
                st.exception(e)
                st.session_state.messages.append({"role": "assistant", "content": f"ERROR: {repr(e)}"})




