import os
import streamlit as st
from dotenv import load_dotenv
from functions.utils import (
    load_pg_retriever, recuperar_documentos,
    gerar_prompt_rag, chamar_api_deepseek
)
import html as _html  # para escapar conteúdo das mensagens

# ----- Config -----
load_dotenv(dotenv_path="config/config.env")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY não encontrada. Defina-a em config.env")
st.set_page_config(page_title="Corujita - Chatbot", page_icon="🦉")

# ----- Estado -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🦉 Olá! Eu sou a Corujita, a assistente virtual do GATE! Em que posso ajudar?"}
    ]

# único slot para o chat (sempre repintamos aqui)
if "chat_slot" not in st.session_state:
    st.session_state.chat_slot = st.empty()

# ----- Retriever -----
recuperador = load_pg_retriever(k_default=5)

def processa_resposta(entrada: str) -> str:
    try:
        docs = recuperar_documentos(
            os.environ.get("DEEPSEEK_API_KEY",""),
            entrada,
            recuperador,
            expandir=True, # atenção! expandir cria uma query de busca com 10 sinônimos separados por ponto (em uma query única). Por isso, se usar expandir=True, melhor usar rerank_method = deepseek. 
            k_recall=5,
            k_final=3,
            rerank_method="deepseek", # crossencoder ou deepseek
            min_score_reranking=0.0,
        )
    except Exception as e:
        return f"Erro ao recuperar documentos: {e}"

    hist = st.session_state.messages[-5:]
    prompt = gerar_prompt_rag(entrada, docs, hist)
    resp = chamar_api_deepseek(os.environ.get("DEEPSEEK_API_KEY",""), prompt)
    return resp or "Desculpe, não foi possível obter uma resposta no momento."

# ----- Renderer: pinta todo o chat dentro da moldura fixa -----

def render_chat_html(slot, messages):
    css = """<style>
      .chat-frame {
        height: 75vh; overflow-y: auto; border: 1px solid #DDD;
        border-radius: 12px; padding: 12px; background: #fff;
      }
      .header { display:flex; gap:10px; align-items:center; background:#C9D0DA;
        border-radius:10px; padding:10px; margin-bottom:10px; }
      .header img { width: 42px; }
      .bubble { max-width: 75%; padding: 10px 12px; border-radius: 12px; margin: 8px 0; }
      .assistant { background:#E6E8F0; color:#000; text-align:left; }
      .user { background:#B9DDC3; color:#000; margin-left:auto; text-align:right; }
      .label { font-weight:700; margin-right:6px; }
    </style>"""

    header = (
        '<div class="header">'
        '<img src="https://raw.githubusercontent.com/pedropaixaomprj/Projeto-Corujita/e266fab5c04a112b363a09c8da96f35d1270b06e/imgs/corujita_transparente.png">'
        '<div><div style="font-weight:700;">Corujita</div>'
        '<div style="font-size:0.95rem;">Assistente virtual</div></div>'
        '</div>'
    )

    parts = ['<div class="chat-frame">', header]

    for m in messages:
        # Escapa HTML do conteúdo e mantém quebras de linha
        text = _html.escape(m["content"]).replace("\n", "<br>")
        if m["role"] == "assistant":
            parts.append(f'<div class="bubble assistant"><span class="label">Chatbot:</span>{text}</div>')
        else:
            parts.append(f'<div class="bubble user"><span class="label">Você:</span>{text}</div>')

    parts.append('</div>')  # fecha chat-frame

    # Renderiza tudo de uma vez; agora nada vira bloco de código
    slot.markdown(css + "".join(parts), unsafe_allow_html=True)


# ----- Trata pergunta vinda da Home (1x) -----
pending = st.session_state.get("pending_user_message", "")
if pending:
    st.session_state["pending_user_message"] = ""
    st.session_state.messages.append({"role": "user", "content": pending})
    # mostra "Processando…" já dentro da moldura
    st.session_state.messages.append({"role": "assistant", "content": "Processando…"})
    render_chat_html(st.session_state.chat_slot, st.session_state.messages)

    # calcula e substitui
    ans = processa_resposta(pending)
    st.session_state.messages[-1]["content"] = ans
    render_chat_html(st.session_state.chat_slot, st.session_state.messages)

# ----- Primeiro paint do chat (uma vez por execução) -----
render_chat_html(st.session_state.chat_slot, st.session_state.messages)

# ----- Input (chat_input fica fora da moldura, sem fantasma) -----
user_text = st.chat_input("Digite sua pergunta")
if user_text:
    # adiciona usuário
    st.session_state.messages.append({"role": "user", "content": user_text})
    # mostra "Processando…" imediatamente
    st.session_state.messages.append({"role": "assistant", "content": "Processando…"})
    render_chat_html(st.session_state.chat_slot, st.session_state.messages)

    # calcula e substitui
    answer = processa_resposta(user_text)
    st.session_state.messages[-1]["content"] = answer
    render_chat_html(st.session_state.chat_slot, st.session_state.messages)
