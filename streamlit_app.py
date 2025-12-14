import streamlit as st
from legal_core import retrieve, build_context, generate_answer, rows_to_sources

st.set_page_config(page_title="Asistente Legal", page_icon="⚖️", layout="centered")
st.title("⚖️ Asistente Legal (Vector Search + OpenAI)")

with st.sidebar:
    st.header("Opciones")
    show_sources = st.checkbox("Mostrar fuentes", value=True)
    show_context = st.checkbox("Mostrar contexto (debug)", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Escribí tu consulta legal…")

if prompt:
    # Mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesar respuesta
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
                err = f"ERROR: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
