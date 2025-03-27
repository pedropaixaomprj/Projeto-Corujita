import streamlit as st

# Título da página
st.set_page_config(page_title="Corujita - funcionamento", page_icon="🦉")


st.title("Como o Corujita funciona?")


st.markdown("""
<div style='text-align: justify;'>
O Corujita segue um fluxo de trabalho dividido em duas etapas principais, baseadas na arquitetura RAG (Retrieval Augmented Generation) para construção de chatbots. Essas etapas consistem em pré-processamento e inferência.
<br><br>
Na fase de pré-processamento, inicialmente, ocorre a aquisição e curadoria de um dataset de referência, que consiste em um conjunto de pares de perguntas e respostas mais comuns recebidas pelo GATE com relação aos seus sistemas. Este dataset é submetido a um processo de limpeza e validação para garantir a qualidade e relevância das informações. Em seguida, um Large Language Model (LLM) é utilizado para expandir semanticamente o dataset, gerando variações das perguntas originais, o que aumenta a cobertura lexical e a robustez do sistema. As perguntas do dataset são então convertidas em vetores de embeddings, utilizando um modelo pré-treinado de embeddings de documentos, que capturam o significado semântico das perguntas. Por fim, os embeddings gerados são armazenados em um índice vetorial FAISS, permitindo a busca eficiente de informações relevantes.
<br><br>Na etapa de inferência, o sistema recebe a consulta do usuário através da interface do chatbot. A pergunta é convertida em um vetor de embeddings, utilizando o mesmo modelo utilizado no pré-processamento. Uma busca de similaridade é realizada no FAISS para recuperar os embeddings mais similares, juntamente com seus documentos de origem. Selecionamos todas as perguntas do dataset de referência que estão "próximas o suficiente" do prompt do usuário, a partir de um limiar de distância pré-definido. As informações recuperadas e o histórico recente da conversa são adicionadas ao contexto da pergunta original do usuário, gerando um prompt enriquecido para a LLM. Esse prompt final é enviado para a API da LLM DeepSeek, gerando uma resposta que é então exibida ao usuário através da interface do chatbot.
<br><br>O esquema abaixo resume as diferentes etapas da implementação do chatbot. <br><br>
</div>
""", unsafe_allow_html=True)


st.image("imgs/esquema_resumo_corujita.png")
