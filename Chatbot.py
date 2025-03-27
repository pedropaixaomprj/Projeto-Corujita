# MainPage.py (or app.py)
import os
import streamlit as st
import pickle
from pathlib import Path
from dotenv import load_dotenv
# import torch

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
from sentence_transformers import SentenceTransformer

# Import everything from utils.py
from functions.utils import (
    chamar_api_deepseek,
    expandir_query_com_llm,
    recuperar_documentos,
    gerar_prompt_rag,
    render_chat_history
)


# Avoid conflicts with Torch modules if needed
# torch.classes.__path__ = []

# Load environment variables
# Using Streamlit community:
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

# If there is a file config.env in the folder config
# load_dotenv(Path("config/config.env"))
# DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# if not DEEPSEEK_API_KEY:
#     raise ValueError("DEEPSEEK_API_KEY não encontrada. Por favor, defina-a nas variáveis de ambiente.")

# Load the VectorStore index from pickle
# with open("faiss_index/vectorstore.pkl", "rb") as arquivo:
#    vectorstore = pickle.load(arquivo)
#    recuperador = VectorStoreRetriever(vectorstore=vectorstore)

###############################################################################################
##### NÃO ESTÁ CARREGANDO O MODELO MULTILÍNGUE CORRETAMENTE: ESTÁ CARREGANDO MEAN_POOLING #####
###############################################################################################

# Loading embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Loading from faiss_index
vectorstore = FAISS.load_local("./faiss_index",
                                embedding_model,
                                allow_dangerous_deserialization=True) # Nesse caso, não é "dangerous" porque eu mesmo gerei os indexes

recuperador = VectorStoreRetriever(vectorstore=vectorstore)

# Streamlit UI
st.set_page_config(page_title="Corujita - Chatbot", page_icon="🦉")

def main():
    # Layout header with image and title
    co1, co2, co3, co4, co5 = st.columns(5)
    with co3:
        st.image("imgs/logo-coruja.svg", width=80)

    st.markdown(
        """
        <h2 style='text-align: center;'>Corujita - Chatbot Assistente do GATE</h2>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state for chat history and processing flag
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "🦉 Olá! Em que posso ajudar? Você tem alguma pergunta?"
        }]
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Create an empty container to hold the chat history
    chat_container = st.empty()

    # Render the initial chat history
    render_chat_history(chat_container, st.session_state.chat_history, st.session_state.processing)

    # Form for user input
    with st.form(key="chat_form"):
        entrada_usuario = st.text_input("Digite sua pergunta:")
        submit_button = st.form_submit_button("Enviar", disabled=st.session_state.processing)

    # If the form is submitted and there's input, start processing
    if submit_button and entrada_usuario and not st.session_state.processing:
        st.session_state.processing = True
        # Add the user question to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": entrada_usuario
        })

        # Re-render chat history (to show the user's question + "processando")
        render_chat_history(chat_container, st.session_state.chat_history, st.session_state.processing)

        # Process the answer
        try:
            documentos = recuperar_documentos(
                DEEPSEEK_API_KEY,
                entrada_usuario,
                recuperador,
                expandir=False,
                k=3
            )
        except Exception as e:
            st.error(f"Erro ao recuperar documentos: {e}")
            st.session_state.processing = False
            st.stop()

        # Optionally, limit the history used in the RAG prompt
        MAX_HISTORICO = 5
        historico_chat = st.session_state.chat_history[-MAX_HISTORICO:]

        # Generate final prompt
        prompt_final = gerar_prompt_rag(entrada_usuario, documentos, historico_chat)

        # Call the DeepSeek API for the final answer
        resposta_final = chamar_api_deepseek(DEEPSEEK_API_KEY, prompt_final)
        if not resposta_final:
            resposta_final = "Desculpe, não foi possível obter uma resposta no momento."

        # Append the assistant's answer to the chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": resposta_final
        })
        st.session_state.processing = False

        # Re-render the chat history with the new answer
        render_chat_history(chat_container, st.session_state.chat_history, st.session_state.processing)

if __name__ == "__main__":
    main()
