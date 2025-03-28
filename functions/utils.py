# utils.py

import requests
import pickle
from langchain.vectorstores.base import VectorStoreRetriever
import torch

# (Optional) Avoid conflicts with Torch modules
torch.classes.__path__ = []

# Load the VectorStore index from SentenceTransformers
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return VectorStoreRetriever(vectorstore=vectorstore)

def chamar_api_deepseek(chave_api, texto_entrada):
    """
    Realiza a chamada à API DeepSeek para obter a resposta do modelo.
    """
    url = "https://api.deepseek.com/v1/chat/completions"  # Ajustar URL se necessário
    cabecalhos = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {chave_api}"
    }
    dados = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": texto_entrada}]
    }
    
    try:
        resposta = requests.post(url, headers=cabecalhos, json=dados, verify=False)
        resposta.raise_for_status()
        resposta_json = resposta.json()
        if "choices" in resposta_json:
            return resposta_json["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        print(f"Erro na chamada da API DeepSeek: {e}")
        return None


def expandir_query_com_llm(chave_api, query_original):
    """
    Expande a consulta original utilizando a API DeepSeek.
    """
    prompt = f"""
Dado o contexto de um assistente virtual para o sistema Nova SAT do Ministério Público do Estado do Rio de Janeiro (MPRJ), 
expanda a seguinte pergunta do usuário gerando 10 sinônimos, reformulações e variações que poderiam ter sido feitas por outros usuários no passado. 

O objetivo é melhorar a recuperação da resposta correta, garantindo que perguntas semelhantes sejam identificadas como equivalentes. 
Priorize termos técnicos e expressões usadas no contexto jurídico e administrativo do MPRJ.

Pergunta original: "{query_original}"

Exemplo de query: "Estou com problema no login".
Exemplo de resposta (resumido): "Estou com problema no login. Não consigo logar na página. Não consigo entrar na nova SAT. O login não funciona. O que fazer quando o login está bugado?.

Responda apenas com as expressões-sinônimos, separadas por ponto. Não acrescente nada mais além disso. Não acrescente números indicadores da lista.
    """
    resposta = chamar_api_deepseek(chave_api, prompt)
    if resposta:
        return resposta
    return query_original


def recuperar_documentos(chave_api, query_original, recuperador, expandir=False, k=5, limiar=0.3):
    """
    Recupera documentos relevantes utilizando similaridade semântica e, opcionalmente, expansão da consulta.
    """
    # Expande a query, se necessário
    query_final = expandir_query_com_llm(chave_api, query_original) if expandir else query_original

    # Obtém as k referências com seus respectivos scores
    referencias_com_score = recuperador.vectorstore.similarity_search_with_score(query_final, k=k)
    print(f'\nReferências com score: {referencias_com_score}\n')

    # Dicionário para armazenar respostas únicas com base no score
    respostas_unicas = {}
    
    # Filtra e armazena apenas os documentos com score acima do limiar
    for doc, score in referencias_com_score:
        if score >= limiar:  # Ajuste se o score for inverso (distância) ou direto (similaridade)
            resposta_doc = doc.metadata.get("Resposta")
            # Se ainda não existe essa resposta_doc ou encontramos score maior agora, armazena
            if resposta_doc and (resposta_doc not in respostas_unicas or respostas_unicas[resposta_doc][1] < score):
                respostas_unicas[resposta_doc] = (doc, score)

    documentos_relevantes = [doc for doc, score in respostas_unicas.values()]
    return documentos_relevantes


def gerar_prompt_rag(pergunta, documentos=None, historico_chat=None):
    """
    Gera um prompt que combina a pergunta do usuário com o contexto extraído dos documentos 
    recuperados e, opcionalmente, com parte do histórico do chat.
    """
    # Process documents context
    contexto_docs = ""
    if documentos:
        partes_contexto = []
        for doc in documentos:
            conteudo = doc.metadata.get("Resposta", "Não foi encontrada informação relevante")
            partes_contexto.append(conteudo)
        contexto_docs = "\n\n".join(partes_contexto)
    
    # Process chat context
    if historico_chat:
        partes_chat = []
        for mensagem in historico_chat:
            if mensagem["role"] == "assistant":
                partes_chat.append("Chatbot: " + mensagem["content"])
            else:
                partes_chat.append("Usuário: " + mensagem["content"])
        historico_chat = "\n".join(partes_chat)

    prompt = f"""
Você é um assistente virtual especialista do sistema Nova SAT.
Você receberá um histórico de conversas com o usuário.
Utilize estritamente as informações obtidas de documentos a seguir como referência para pergunta do usuário de forma clara, detalhada e objetiva.
Não mencione que sua resposta foi construída com base nessas informações.

Informações possivelmente relevantes obtidas de documentos:
{contexto_docs}

Histórico da conversa
#######################
{historico_chat}
#######################

Mensagem do usuário:
{pergunta}

Caso a mensagem do usuário seja small-talk, responda a small-talk apropriadamente de maneira formal e pergunte qual é a dúvida que a pessoa deseja tirar.
Caso seja uma pergunta e não encontre nenhuma informação relevante, retorne: 'Entre em contato com a secretaria do GATE por meio do Microsoft Teams para maiores informações.'
"""
    print(prompt)
    return prompt


def render_chat_history(container, chat_history, processing=False):
    """
    Renderiza todo o histórico de chat dentro de um container do Streamlit.
    """
    chat_history_html = (
        '<div style="height: 400px; overflow-y: auto; padding: 5px; border: 1px solid #ccc;">'
    )

    for mensagem in chat_history:
        if mensagem["role"] == "assistant":
            chat_history_html += f'''
                <div style="text-align: left; background-color: #E8F0FE;
                             color: #0D3367; border-radius: 10px; padding: 8px; 
                             margin: 5px 0; max-width: 70%;">
                    <strong>Chatbot:</strong> {mensagem['content']}
                </div>
            '''
        else:
            chat_history_html += f'''
                <div style="text-align: right; background-color: #DFF8E1;
                             color: #333333; border-radius: 10px; padding: 8px;
                             margin: 5px 0; max-width: 70%; margin-left: auto;">
                    <strong>Você:</strong> {mensagem['content']}
                </div>
            '''

    if processing:
        chat_history_html += '''
            <div style="text-align: left; background-color: #E8F0FE;
                         color: #0D3367; border-radius: 10px; padding: 8px;
                         margin: 5px 0; max-width: 70%;">
                <strong>Chatbot:</strong> Processando resposta...
            </div>
        '''

    chat_history_html += '</div>'
    container.html(chat_history_html)
