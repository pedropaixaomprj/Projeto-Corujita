# utils.py

import os
import re
import json
import time
import pickle
import datetime
from typing import List, Tuple, Optional, Dict

import requests
import torch
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_core.documents import Document

# (Optional) Avoid conflicts with Torch modules
torch.classes.__path__ = []

# Carrega o .env uma vez no carregamento do módulo
load_dotenv(dotenv_path="config/config.env")

#### ANOTHER OPTION IS TO DEFINE a config.py file
# putting all config loading/validation there and importing it everywhere

EMBED_URL = os.getenv("EMBED_URL", "https://go-llm.mprj.mp.br/st/embed")
EMBED_HEADERS = {
    "Accept": os.getenv("EMBED_ACCEPT", "application/json"),
    "Content-Type": os.getenv("EMBED_CONTENT_TYPE", "application/json;charset=UTF-8"),
    "Accept-Encoding": os.getenv("EMBED_ACCEPT_ENCODING", "gzip,deflate"),
}
EMBED_TIMEOUT = int(os.getenv("EMBED_REQUEST_TIMEOUT", "60"))
EMBED_RETRY_MAX = int(os.getenv("EMBED_RETRY_MAX", "3"))
EMBED_RETRY_BACKOFF = float(os.getenv("EMBED_RETRY_BACKOFF", "1.5"))

# Lê as variáveis de ambiente (globalmente)
PGHOST = os.getenv("PGHOST")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
PGPORT = os.getenv("PGPORT", "5432")
TABLE_NAME = os.getenv("PGVECTOR_TABLE", "nlp.faq_embeddings")
DISTANCE = os.getenv("PGVECTOR_DISTANCE", "cosine").lower()

# --- Lê e valida variáveis obrigatórias ---
required_vars = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise ValueError(
        f"As seguintes variáveis obrigatórias não foram definidas no .env: {', '.join(missing)}"
    )

# ========== LOGGING HELPER ==========
LOG_ENABLED = os.getenv("RAG_DEBUG", "1") in ("1", "true", "True", "yes", "YES")

def log(stage: str, msg: str, *, data: Optional[dict] = None):
    """Print estruturado com timestamp, etapa e opcionalmente um dicionário de dados."""
    if not LOG_ENABLED:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = f"[{ts}] [{stage}] {msg}"
    if data is not None:
        try:
            payload = json.dumps(data, ensure_ascii=False)
        except Exception:
            payload = str(data)
        print(f"{base} :: {payload}")
    else:
        print(base)

# ========== RERANK CONFIG ==========
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "RERANK_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
USE_CROSS_ENCODER = os.getenv("RERANK_METHOD", "deepseek").lower() == "crossencoder"

_cross_encoder = None
def _ensure_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # import lazy
        log("RERANK/CE", "Carregando CrossEncoder", data={"model": CROSS_ENCODER_MODEL_NAME})
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder

# ========== DEEPSEEK CALL ==========
def chamar_api_deepseek(chave_api, texto_entrada):
    """
    Realiza a chamada à API DeepSeek para obter a resposta do modelo.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    cabecalhos = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {chave_api}",
    }
    dados = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": texto_entrada}],
    }
    log("DEEPSEEK/CALL", "→ Enviando prompt ao DeepSeek", data={
        "url": url,
        "chars": len(texto_entrada),
        "preview": texto_entrada[:180] + ("…" if len(texto_entrada) > 180 else "")
    })

    try:
        resposta = requests.post(url, headers=cabecalhos, json=dados, verify=False)
        log("DEEPSEEK/CALL", f"← HTTP {resposta.status_code}")
        resposta.raise_for_status()
        resposta_json = resposta.json()
        if "choices" in resposta_json:
            content = resposta_json["choices"][0]["message"]["content"]
            log("DEEPSEEK/CALL", "Conteúdo recebido", data={
                "chars": len(content),
                "preview": content[:180] + ("…" if len(content) > 180 else "")
            })
            return content
        log("DEEPSEEK/CALL", "Resposta sem campo 'choices'")
        return None
    except Exception as e:
        log("DEEPSEEK/ERROR", f"Erro na chamada da API DeepSeek: {e}")
        return None

# ========== QUERY EXPANSION ==========
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
    log("EXPAND", "Expandindo query com LLM", data={"original": query_original})
    resposta = chamar_api_deepseek(chave_api, prompt)
    if resposta:
        log("EXPAND", "Query expandida", data={
            "chars": len(resposta),
            "preview": resposta[:180] + ("…" if len(resposta) > 180 else "")
        })
        return resposta
    log("EXPAND", "Falha na expansão, usando query original")
    return query_original

# ========== RAG CORE ==========
def _make_candidate_text(doc: Document, max_chars: int = 600) -> str:
    """
    Concatena pergunta + (trecho da) resposta para o reranker decidir relevância.
    """
    pergunta = doc.page_content or ""
    resposta = doc.metadata.get("resposta", "") or ""
    txt = f"Pergunta: {pergunta}\nResposta: {resposta}"
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "…"
    return txt

def rerank_with_cross_encoder(
    query: str,
    docs: List[Document],
    top_m: int,
    min_score_reranking: Optional[float] = None  # <-- NOVO: corte mínimo por score (CrossEncoder costuma ~0–1)
) -> List[Tuple[Document, float]]:
    """
    Usa CrossEncoder para reordenar (scores maiores = melhores).
    Retorna [(doc, score)] ordenado desc.
    """
    if not docs:
        return []
    log("RERANK/CE", "Iniciando CrossEncoder", data={
        "k_in": len(docs), "top_m": top_m,
        "model": CROSS_ENCODER_MODEL_NAME,
        "min_score": min_score_reranking
    })
    ce = _ensure_cross_encoder()
    pairs = [(query, _make_candidate_text(d)) for d in docs]
    scores = ce.predict(pairs)  # numpy array
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    try:
        # .tolist() pode não existir em alguns tipos, então garantimos robustez
        scores_list = [float(s) for s in (scores.tolist() if hasattr(scores, "tolist") else list(scores))]
    except Exception:
        scores_list = [float(s) for s in scores]
    log("RERANK/CE", "Scores CE", data={"scores": [round(float(s), 3) for s in scores_list]})

    # --- NOVO: aplicar corte por score mínimo, se fornecido ---
    if min_score_reranking is not None:
        ranked = [(d, float(s)) for (d, s) in ranked if float(s) >= float(min_score_reranking)]
        log("RERANK/CE", "Aplicado min_score_reranking (CE)", data={
            "min_score": min_score_reranking,
            "restantes": len(ranked)
        })

    return ranked[:top_m]

def rerank_with_deepseek(
    chave_api: str,
    query: str,
    docs: List[Document],
    top_m: int,
    min_score_reranking: Optional[float] = None  # <-- NOVO: corte mínimo por score (DeepSeek: 0–100 no prompt)
) -> List[Tuple[Document, float]]:
    """
    Reordena candidatos usando DeepSeek como juiz, chamando chamar_api_deepseek().
    Retorna [(doc, score)] ordenado por score desc, cortado em top_m.
    """
    if not docs:
        return []

    # helper: chave estável p/ deduplicar
    def _doc_key(d: Document):
        # Se houver um ID estável no metadata, prefira-o:
        return d.metadata.get("pergunta_id") or d.metadata.get("id") or d.metadata.get("source") or id(d)

    # 1) Candidatos numerados
    log("RERANK/DS", "Preparando candidatos", data={"k_in": len(docs), "min_score": min_score_reranking})
    candidatos = [{"id": i, "text": _make_candidate_text(d, max_chars=700)} for i, d in enumerate(docs, 1)]

    # 2) Prompt pedindo SOMENTE a lista JSON
    instrucao = (
        "Você é um reranker. Dada a 'query' e uma lista de candidatos, "
        "atribua uma nota de relevância de 0 a 100 para cada candidato considerando a pergunta do usuário. "
        "Seja estrito (penalize duplicatas e respostas vagas). "
        "Responda SOMENTE com um JSON que seja uma LISTA no formato: "
        '[{\"id\": <int>, \"score\": <int>}]. Sem texto adicional.'
    )
    payload_usuario = {"query": query, "candidates": candidatos}
    texto_entrada = instrucao + "\n\n" + json.dumps(payload_usuario, ensure_ascii=False)
    log("RERANK/DS", "Montando prompt p/ DeepSeek", data={
        "query_chars": len(query),
        "cands_chars": sum(len(c["text"]) for c in candidatos)
    })

    # 3) Chamada
    content = chamar_api_deepseek(chave_api, texto_entrada)
    if not content:
        log("RERANK/DS", "Sem conteúdo do DeepSeek, fallback")
        base, step = 100.0, 1.0
        return [(d, base - i * step) for i, d in enumerate(docs)]

    # 4) Parser "tolerante" de JSON
    def _parse_json_loose(txt: str):
        s = txt.strip().strip("` \n")
        if s.lower().startswith("json"):
            s = s[4:].lstrip(": \n")
        # tenta direto
        try:
            return json.loads(s)
        except Exception:
            pass
        # tenta extrair primeiro bloco JSON (prioriza lista)
        m = re.search(r"\[[\s\S]*\]", s) or re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    try:
        log("RERANK/DS", "Parsing resposta do DeepSeek", data={"chars": len(content)})
        data = _parse_json_loose(content)
        if data is None:
            raise ValueError("Não foi possível parsear JSON devolvido pelo LLM.")

        # Aceita lista direta ou dict com 'ranking'
        if isinstance(data, list):
            ranking_list = data
        elif isinstance(data, dict):
            ranking_list = data.get("ranking", [])
        else:
            raise TypeError(f"Formato inesperado do JSON: {type(data)}")

        log("RERANK/DS", "Ranking bruto do LLM", data={"n": len(ranking_list), "preview": ranking_list[:5]})

        # 5) Mapeia ids -> docs e coleta scores
        id2doc = {i + 1: d for i, d in enumerate(docs)}
        ranked: List[Tuple[Document, float]] = []
        for item in ranking_list:
            try:
                if isinstance(item, dict):
                    did = int(item["id"])
                    score = float(item["score"])
                else:
                    did = int(item[0])
                    score = float(item[1])
                if did in id2doc:
                    ranked.append((id2doc[did], score))
            except Exception:
                continue

        # 6) Dedup + complemento
        seen_keys = set()
        dedup_ranked: List[Tuple[Document, float]] = []
        for doc, score in ranked:
            k = _doc_key(doc)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            dedup_ranked.append((doc, score))

        for d in docs:
            k = _doc_key(d)
            if k not in seen_keys:
                dedup_ranked.append((d, 0.0))
                seen_keys.add(k)

        log("RERANK/DS", "Dedup+complemento concluído", data={
            "n_after": len(dedup_ranked),
            "top_scores": [round(float(s), 2) for _, s in dedup_ranked[:min(5, len(dedup_ranked))]]
        })

        # --- NOVO: aplicar corte por score mínimo, se fornecido (antes do sort+top_m) ---
        if min_score_reranking is not None:
            before = len(dedup_ranked)
            dedup_ranked = [(d, float(s)) for (d, s) in dedup_ranked if float(s) >= float(min_score_reranking)]
            log("RERANK/DS", "Aplicado min_score_reranking (DeepSeek)", data={
                "min_score": min_score_reranking,
                "antes": before, "depois": len(dedup_ranked)
            })

        # 7) Ordena por score desc e corta top_m
        dedup_ranked.sort(key=lambda x: float(x[1]), reverse=True)
        final_ranked = dedup_ranked[:max(1, min(top_m, len(dedup_ranked)))]
        log("RERANK/DS", "Ranking final", data={
            "n": len(final_ranked),
            "ids": [d.metadata.get("pergunta_id") for d, _ in final_ranked],
            "scores": [round(float(s), 2) for _, s in final_ranked],
        })
        return final_ranked

    except Exception as e:
        log("RERANK/DS", f"Falha no rerank DeepSeek, fallback", data={"error": str(e)})
        base, step = 100.0, 1.0
        return [(d, base - i * step) for i, d in enumerate(docs)]

def recuperar_documentos(
    chave_api: str,
    query_original: str,
    recuperador,
    expandir: bool = False,
    k_recall: int = 8,
    k_final: int = 3,
    rerank_method: str = None,  # "deepseek" | "crossencoder" | None
    limiar: Optional[float] = None,  # hard filter pela distância do pgvector (menor=melhor)
    min_score_reranking: Optional[float] = None  # <-- NOVO: corte mínimo por score no reranking
) -> List[Document]:
    """
    Pipeline em 2 estágios:
      (1) Recall via pgvector => top k_recall (com filtro opcional por distância 'limiar')
      (2) Reranking => ordena e retorna top k_final
    Anota cada Document final com:
      - metadata['rerank_pos']   (posição 1..N no rerank final)
      - metadata['rerank_score'] (score do rerank; CE ou DeepSeek)
    """
    method = (rerank_method or os.getenv("RERANK_METHOD", "deepseek")).lower()
    log("RAG", "Iniciando recuperação", data={
        "method": method, "k_recall": k_recall, "k_final": k_final,
        "expandir": expandir, "limiar": limiar, "min_score": min_score_reranking
    })

    # 1) Expansão opcional
    query_final = expandir_query_com_llm(chave_api, query_original) if expandir else query_original
    if expandir:
        log("RAG", "Query final após expansão", data={"query_final": query_final})

    # 2) Recall bruto no pgvector
    refs = recuperador.similarity_search_with_score(query_final, k=k_recall)
    if not refs:
        log("RAG", "Nenhuma referência encontrada")
        return []

    # 2a) Filtro opcional por limiar de distância (menor = melhor, pois é cosseno)
    if limiar is not None:
        refs = [(d, dist) for (d, dist) in refs if dist <= float(limiar)]
        log("RAG", "Aplicado filtro por limiar (pgvector)", data={
            "limiar": limiar, "restantes": len(refs)
        })
        if not refs:
            log("RAG", "Nenhuma referência após limiar")
            return []

    docs_raw = [doc for (doc, _dist) in refs]
    log("RAG", "Recall bruto OK", data={"n_docs": len(docs_raw)})

    # 3) Reranking
    if method == "crossencoder":
        ranked_pairs: List[Tuple[Document, float]] = rerank_with_cross_encoder(
            query_final, docs_raw, top_m=k_recall, min_score_reranking=min_score_reranking  # <-- NOVO
        )
    elif method == "deepseek":
        ranked_pairs = rerank_with_deepseek(
            chave_api, query_final, docs_raw, top_m=k_recall, min_score_reranking=min_score_reranking  # <-- NOVO
        )
    else:
        # Sem rerank: usa ordem do pgvector (já por distância ascendente); atribui score 0.0
        ranked_pairs = [(doc, 0.0) for doc in docs_raw]

    log("RAG", "Após rerank", data={
        "n_ranked": len(ranked_pairs),
        "scores": [round(float(s), 3) for _, s in ranked_pairs[:min(10, len(ranked_pairs))]]
    })

    # 3a) Construir mapa doc_key -> (posicao, score) para anotação posterior
    def _doc_key(d: Document):
        # se tiver, use um ID estável seu; senão, cai para id(d)
        return d.metadata.get("pergunta_id") or d.metadata.get("id") or d.metadata.get("source") or id(d)

    pos_map: Dict[object, Tuple[int, float]] = {}
    for idx, (d, s) in enumerate(ranked_pairs, start=1):
        pos_map[_doc_key(d)] = (idx, float(s))

    # 4) Seleção final com deduplicação por resposta (para evitar repetições da mesma FAQ)
    respostas_vistas: Dict[str, bool] = {}
    final_ranked: List[Tuple[Document, float]] = []
    for doc, score in ranked_pairs:
        resp = (doc.metadata.get("resposta") or "").strip()
        if resp and not respostas_vistas.get(resp):
            final_ranked.append((doc, float(score)))
            respostas_vistas[resp] = True
        if len(final_ranked) >= k_final:
            break

    # Se por acaso filtrou demais, completa com o restante já reordenado
    if len(final_ranked) < k_final:
        for doc, score in ranked_pairs:
            if all(doc is not d for d, _ in final_ranked):
                final_ranked.append((doc, float(score)))
            if len(final_ranked) >= k_final:
                break

    # 5) Cortar, anotar metadados e retornar apenas os Documents
    final_ranked = final_ranked[:k_final]

    # Ordene novamente por score desc só por garantia (caso a dedup altere ordem)
    final_ranked.sort(key=lambda x: float(x[1]), reverse=True)

    # Anotar posição (1..N) e score no metadata de cada doc final
    for idx, (doc, score) in enumerate(final_ranked, start=1):
        k = _doc_key(doc)
        pos, sc = pos_map.get(k, (idx, score))
        doc.metadata["rerank_pos"] = pos
        doc.metadata["rerank_score"] = sc

    log("RAG", "Documentos finais", data={
        "n_final": len(final_ranked),
        "ids": [doc.metadata.get("pergunta_id") for doc, _ in final_ranked],
        "pos": [doc.metadata.get("rerank_pos") for doc, _ in final_ranked],
        "scores": [round(float(s), 3) for _, s in final_ranked],
    })

    # Retorna somente os Documents, porque o restante do seu pipeline espera List[Document]
    return [doc for doc, _ in final_ranked]

def gerar_prompt_rag(
    pergunta: str,
    documentos: Optional[List[Document]] = None,
    historico_chat: Optional[List[dict]] = None,
    *,
    show_scores: bool = True,
    show_pg_distance: bool = False,
    max_doc_chars: int = 1200,   # truncate each doc context
    max_total_chars: int = 18000 # safety cap for the whole prompt
) -> str:
    """
    Build a RAG prompt with:
      - Markdown section headers
      - Each context item includes [#pos | score=?] and Q/A
      - Chat history rendered as "Usuário:" / "Chatbot:"
      - Final line '### Sua resposta:' to anchor generation
    """
    def _truncate(s: str, n: int) -> str:
        s = s.strip() if s else ""
        return (s[:n] + "…") if len(s) > n else s

    # ---------- Context (Q + A) ----------
    partes_contexto = []
    if documentos:
        for i, doc in enumerate(documentos, start=1):
            pergunta_doc = _truncate(getattr(doc, "page_content", "") or "", max_doc_chars // 2)
            resposta_doc = _truncate((doc.metadata.get("resposta") or ""), max_doc_chars)
            pos = doc.metadata.get("rerank_pos", i)
            score = doc.metadata.get("rerank_score", None)
            pgd = doc.metadata.get("pg_distance", None) if show_pg_distance else None

            header = f"[#{pos}"
            if show_scores and (score is not None):
                try:
                    header += f" | score={float(score):.2f}"
                except Exception:
                    header += f" | score={score}"
            if pgd is not None:
                try:
                    header += f" | dist={float(pgd):.4f}"
                except Exception:
                    header += f" | dist={pgd}"
            header += "]"

            partes_contexto.append(
                f"{header}\n"
                f"Pergunta: {pergunta_doc}\n"
                f"Resposta: {resposta_doc}"
            )

    contexto_docs = "\n\n".join(partes_contexto) if partes_contexto else "(nenhum documento relevante recuperado)"

    # ---------- Chat history ----------
    hist_lines = []
    if historico_chat:
        for msg in historico_chat:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                hist_lines.append(f"Chatbot: {content}")
            else:
                hist_lines.append(f"Usuário: {content}")
    historico_txt = "\n".join(hist_lines) if hist_lines else "(sem histórico)"

    # ---------- Assemble prompt ----------
    prompt = (
        "Você é um assistente virtual especialista do Grupo de Apoio Técnico Especializado (GATE) do Ministério Público do Estado do Rio de Janeiro (GATE/MPRJ).\n"
        "O sistema Nova SAT é utilizado para registrar solicitações de análises técnicas por técnicos periciais do GATE/MPRJ \n\n"
        "Responda apenas perguntas relacionadas a processos, solicitações ou procedimentos do MPRJ.\n"
        "Use as informações a seguir — obtidas de documentos internos — como base factual para sua resposta. "
        "Formule a resposta de forma clara, completa e objetiva. "
        "Não mencione explicitamente que o conteúdo foi extraído dos documentos.\n\n"
        "### Informações possivelmente relevantes\n"
        f"{contexto_docs}\n\n"
        "### Histórico da conversa\n"
        f"{historico_txt}\n\n"
        "### Mensagem do usuário\n"
        f"{pergunta}\n\n"
        "Se a mensagem do usuário for uma saudação ou small-talk, relembre sua função (asssistente virtual do GATE/MPRJ) e pergunte qual é a dúvida técnica.\n"
        "Se não houver informação suficiente nos documentos para responder, diga:\n"
        "\"Entre em contato com a secretaria do GATE por meio do Microsoft Teams para maiores informações.\"\n\n"
        "### Sua resposta:"
    )

    # ---------- Safety cap ----------
    if len(prompt) > max_total_chars:
        prompt = prompt[:max_total_chars] + "…"

    # Optional: structured logging if you kept `log()`
    try:
        log("PROMPT", "Prompt RAG montado", data={
            "docs": len(documentos or []),
            "chars": len(prompt),
            "pos": [d.metadata.get("rerank_pos") for d in (documentos or [])],
            "scores": [d.metadata.get("rerank_score") for d in (documentos or [])],
            "prompt_final": f"\n \n {prompt} \n \n"
        })
    except Exception:
        pass

    return prompt

# ========== PGVECTOR & EMBEDDINGS ==========
def fetch_embedding(text: str, session: Optional[requests.Session] = None) -> List[float]:
    if session is None:
        session = requests.Session()
    payload = {"text": text}
    last_err = None
    log("EMBED", "Solicitando embedding", data={"chars": len(text), "preview": text[:120] + ("…" if len(text) > 120 else "")})
    for attempt in range(EMBED_RETRY_MAX):
        try:
            t0 = time.time()
            r = session.post(EMBED_URL, headers=EMBED_HEADERS, json=payload, timeout=EMBED_TIMEOUT, verify=False)
            dt = round((time.time() - t0)*1000)
            log("EMBED", f"HTTP {r.status_code} em {dt} ms (tentativa {attempt+1}/{EMBED_RETRY_MAX})")
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise ValueError(f"Resposta inesperada do embed: {type(data)}")
            vec = [float(x) for x in data]
            log("EMBED", "Embedding OK", data={"dim": len(vec)})
            return vec
        except Exception as e:
            last_err = e
            log("EMBED/WARN", f"Falha na tentativa {attempt+1}", data={"error": str(e)})
            if attempt < EMBED_RETRY_MAX - 1:
                time.sleep(EMBED_RETRY_BACKOFF ** (attempt + 1))
    raise RuntimeError(f"Falha ao obter embedding. Último erro: {last_err}")

@st.cache_resource
def get_pg_conn():
    """
    Retorna uma conexão persistente com o PostgreSQL, cacheada pelo Streamlit.
    Garante que a conexão seja criada apenas uma vez por sessão.
    """
    os.environ["CURL_CA_BUNDLE"] = os.getenv("CURL_CA_BUNDLE", "")
    os.environ["REQUESTS_CA_BUNDLE"] = os.getenv("REQUESTS_CA_BUNDLE", "")

    try:
        conn = psycopg2.connect(
            host=PGHOST,
            database=PGDATABASE,
            user=PGUSER,
            password=PGPASSWORD,
            port=PGPORT,
        )
        conn.autocommit = True
        print(f"[INFO] Conectado ao PostgreSQL em {PGHOST}:{PGPORT} ({PGDATABASE})")
        return conn
    except Exception as e:
        raise ConnectionError(f"Falha ao conectar ao PostgreSQL: {e}")

class PgVectorSimpleRetriever:
    """
    Retriever simples para pgvector. Implementa similarity_search_with_score(query, k).
    Retorna lista de (Document, distance). Menor distância = melhor.
    """

    def __init__(self, k: int = 5, table_name: str = None):
        self.k = k
        self._conn = get_pg_conn()
        self._table = table_name or os.getenv("PGVECTOR_TABLE", "nlp.faq_embeddings")

    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        k = k or self.k
        log("PGV/RECALL", "Buscando no pgvector", data={"k": k, "table": self._table})
        qvec = fetch_embedding(query)
        qlit = "[" + ",".join(f"{float(x):.8f}" for x in qvec) + "]"

        sql = f"""
            SELECT 
                pergunta_id, pergunta_var_id, pergunta, resposta, ultima_atualizacao,
                embedding_st <=> %s::vector AS distance
            FROM {self._table}
            ORDER BY distance ASC
            LIMIT %s;
        """
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (qlit, k))
            rows = cur.fetchall()

        results: List[Tuple[Document, float]] = []
        for r in rows:
            metadata = {
                "pergunta_id": r["pergunta_id"],
                "pergunta_var_id": r["pergunta_var_id"],
                "ultima_atualizacao": str(r["ultima_atualizacao"]) if r["ultima_atualizacao"] else None,
                "resposta": r["resposta"],
            }
            doc = Document(page_content=r["pergunta"], metadata=metadata)
            results.append((doc, float(r["distance"])))
        log("PGV/RECALL", "Itens recuperados", data={
            "n": len(results),
            "distances": [round(d, 4) for _, d in results],
            "ids": [doc.metadata.get("pergunta_id") for doc, _ in results]
        })
        return results

@st.cache_resource
def load_pg_retriever(k_default: int = 5):
    return PgVectorSimpleRetriever(k=k_default, table_name=os.getenv("PGVECTOR_TABLE", "nlp.faq_embeddings"))
