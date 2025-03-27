# Corujita - Chatbot Assistente do GATE

## Visão Geral

O **Corujita** é um chatbot construído sobre a arquitetura **RAG (Retrieval Augmented Generation)** para auxiliar no atendimento relacionado ao sistema **Nova SAT** do Ministério Público do Estado do Rio de Janeiro (MPRJ). Ele opera em duas etapas principais:

1. **Pré-Processamento**  
   - **Aquisição e Curadoria de Dados**  
     Um dataset de perguntas e respostas frequentes recebidas pelo GATE é montado e limpo para garantir a qualidade do conteúdo.  
   - **Expansão das Perguntas**  
     Um modelo de linguagem (LLM) gera variações das perguntas originais, ampliando a cobertura lexical e robustez do sistema.  
   - **Criação de Embeddings e Indexação FAISS**  
     As perguntas são transformadas em vetores de embeddings e indexadas em um índice vetorial (FAISS) para possibilitar a recuperação eficiente de documentos relevantes.

2. **Inferência**  
   - **Busca de Similaridade**  
     Quando o usuário faz uma consulta, a pergunta é convertida em um vetor de embeddings e comparada às perguntas do dataset (armazenadas no índice FAISS). As que forem consideradas “próximas o suficiente” (com base em um limiar de distância ou similaridade) são selecionadas.  
   - **Montagem de Prompt Enriquecido**  
     As informações encontradas (perguntas/respostas similares) e o histórico recente da conversa são adicionadas ao contexto. Este contexto gerado é enviado como prompt para a API **DeepSeek**, que retorna a resposta final.  
   - **Resposta ao Usuário**  
     Por fim, o chatbot exibe a resposta final ao usuário.

### Arquivos Importantes

- **`Chatbot.py`**: Implementa a interface do chatbot com Streamlit, além de funções para recuperação dos documentos (FAISS), chamada à API da LLM DeepSeek e geração do prompt final.  
- **`pages/Como Funciona.py`: Contém o código referente à segunda página.
- **`data/dataset_corujita_expansao.csv`**: Dataset de referência após aumento via LLMs
- **`faiss_index/vectorstore.pkl`**: Arquivo de índice FAISS persistido, contendo os embeddings do dataset de referência.  
- **`config/config.env`**: Contém variáveis de ambiente, incluindo a chave da API DeepSeek (`DEEPSEEK_API_KEY`).  
- **`requirements.txt`**: Lista de dependências necessárias para executar o projeto.

## Pré-Requisitos

1. **Python 3.8+**  
2. **Virtualenv** (opcional, porém recomendado).  
3. **Instalar dependências**:  
   ```bash
   pip install -r requirements.txt

## Fluxo de Trabalho

### Carregamento do Índice
O arquivo `vectorstore.pkl` já contém o índice FAISS gerado durante a etapa de pré-processamento.

### Recebimento da Pergunta (Interface)
O usuário digita a consulta no campo de texto do chatbot e envia.

### Recuperação de Documentos
A pergunta é convertida em um vetor de embeddings, e o índice FAISS é consultado para encontrar as correspondências mais próximas.

### Geração de Prompt Enriquecido
A pergunta do usuário, o histórico da conversa e as respostas de documentos relevantes são compilados em um único prompt.

### Chamada à API DeepSeek
O prompt enriquecido é enviado à API da DeepSeek para gerar a resposta final com base no contexto fornecido.

### Resposta ao Usuário
O chatbot exibe a resposta proveniente da API.

---

## Customizações e Ajustes

### Tamanho do Histórico
A variável `MAX_HISTORICO` no código controla quantas mensagens recentes são enviadas como contexto no prompt.

### Limiar de Similaridade
Ajuste a variável `limiar` na função `recuperar_documentos` para determinar quão “próximos” os documentos devem estar da consulta do usuário.

### Expansão de Consulta
A função `expandir_query_com_llm` pode ser habilitada/desabilitada para gerar sinônimos e reformulações, dependendo da necessidade.

---

## Possíveis Erros ou Dificuldades

### Chave de API Inexistente ou Inválida
Verifique o valor de `DEEPSEEK_API_KEY` no arquivo `config.env`.

### Versões de Pacotes Incompatíveis
Caso encontre erros na instalação de pacotes ou importações, experimente ajustar ou fixar versões no `requirements.txt`.

### Execução em Windows
Ao utilizar FAISS no Windows, pode ser necessário instalar através do conda-forge ou usar uma versão pré-compilada.

---

## Contribuindo
Sinta-se à vontade para abrir Issues ou Pull Requests para relatórios de bugs ou funcionalidades adicionais.  
Sempre inclua comentários explicando suas alterações e, se possível, exemplos de uso.

