# MainPage.py (or app.py)
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# importa funções do utils.py (já adaptado para pgvector)
from functions.utils import (
    load_pg_retriever,
    recuperar_documentos,
    gerar_prompt_rag,
    chamar_api_deepseek,
    render_chat_history,
)

# Variáveis de ambiente
load_dotenv(dotenv_path="config/config.env")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY não encontrada. Por favor, defina-a nas variáveis de ambiente.")

st.set_page_config(page_title="Corujita - Chatbot", page_icon="🦉")

# Carrega o retriever baseado no pgvector
recuperador = load_pg_retriever(k_default=5)

def main():
    # Wrap everything in a container so they load together
    with st.container():

        # Initialize chat history if not in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{
                "role": "assistant",
                "content": "🦉 Olá! Eu sou a Corujita, a assistente virtual do GATE! Em que posso ajudar?"
            }]
        if "processing" not in st.session_state:
            st.session_state.processing = False

        # Create a chat container inside the same container context
        chat_container = st.empty()

        # If there's a pending user message from the HomePage, add it and process immediately
        pending = st.session_state.get("pending_user_message", "")

        if pending:
            st.session_state.chat_history.append({
                "role": "user",
                "content": pending
            })
            # Clear the pending message so it doesn't reprocess on reload
            st.session_state["pending_user_message"] = ""
            st.session_state.processing = True
            render_chat_history(chat_container, st.session_state.chat_history, st.session_state.processing)

            processa_resposta(pending, chat_container)

        # User input form for new messages
        with st.form(key="chat_form"):
            entrada_usuario = st.text_input("Digite sua pergunta:")
            submit_button = st.form_submit_button("Enviar", disabled=st.session_state.processing)

        if submit_button and entrada_usuario and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.chat_history.append({
                "role": "user",
                "content": entrada_usuario
            })
            render_chat_history(chat_container, st.session_state.chat_history, st.session_state.processing)
            processa_resposta(entrada_usuario, chat_container)

def processa_resposta(entrada_usuario, chat_container):
    try:
        # IMPORTANTE: no pgvector, o "score" retornado pelo retriever é distância (menor é melhor).
        # Se definirmos um limiar, importante valiar empiricamente! Se necessário, podemos remover o filtro no utils.
        documentos = recuperar_documentos(
            os.environ.get("DEEPSEEK_API_KEY", ""),
            entrada_usuario,
            recuperador,
            expandir=False,
            k=3
        )
    except Exception as e:
        st.error(f"Erro ao recuperar documentos: {e}")
        st.session_state.processing = False
        st.stop()
    
    MAX_HISTORICO = 5
    historico_chat = st.session_state.chat_history[-MAX_HISTORICO:]
    prompt_final = gerar_prompt_rag(entrada_usuario, documentos, historico_chat)
    resposta_final = chamar_api_deepseek(os.environ.get("DEEPSEEK_API_KEY", ""), prompt_final)
    
    if not resposta_final:
        resposta_final = "Desculpe, não foi possível obter uma resposta no momento."
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": resposta_final
    })

    st.session_state.processing = False
    render_chat_history(chat_container, st.session_state.chat_history, st.session_state.processing)

if __name__ == "__main__":
    main()