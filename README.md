# python-langchain-rag-advanced

Exemplos de RAG com LangChain para consultar os PDFs de cartões de crédito em
`docs/creditcards/` e comparar as respostas de quatro cadeias:

- `rewrite`: reformula a pergunta antes da busca.
- `multi-query`: gera perguntas alternativas e reúne os documentos encontrados.
- `hyde`: gera uma resposta hipotética para orientar a busca.
- `ollama`: usa embeddings e um modelo de linguagem locais.

## Preparação

Instale as dependências no seu ambiente Python:

```powershell
python -m pip install -r requirements.txt
```

Crie um `.env` na raiz seguindo `.env.example` e preencha `OPENAI_API_KEY`.
As configurações do LangSmith são opcionais.

Para usar a cadeia Ollama, mantenha o serviço em execução e disponibilize os modelos
`bge-m3:567m` e `gemma3:4b`:

```powershell
ollama pull bge-m3:567m
ollama pull gemma3:4b
```

Execute os comandos abaixo a partir da raiz do projeto. Os módulos das cadeias
usam underscores nos nomes para permitir a importação pelo Python.

## Documentos e índice compartilhados

O fluxo é: **PDFs → chunks → índice FAISS → cadeia → avaliação**.

`src/rag/documents.py` concentra o carregamento dos PDFs e a divisão com o
tokenizer `BAAI/bge-m3`, tamanho de 1250 tokens e sobreposição de 150. Os chunks
são reutilizados em memória durante a mesma execução.

`src/rag/vectorstore.py` cria e salva o índice FAISS no primeiro uso. Nas próximas
execuções, carrega o índice existente, incluindo os documentos dos chunks, sem
dividir os PDFs novamente nem gerar de novo os embeddings dos documentos.

Há dois índices, porque os modelos de embeddings são diferentes:

| Cadeias | Embeddings | Diretório do índice |
| --- | --- | --- |
| Rewrite, multi-query e HyDE | OpenAI `text-embedding-3-small` | `.cache/rag/openai-text-embedding-3-small/` |
| Ollama | Ollama `bge-m3:567m` | `.cache/rag/ollama-bge-m3-567m/` |

Os documentos e parâmetros de divisão são considerados fixos. O fluxo apenas
cria ou carrega o índice. Não há arquivo JSON adicional para os chunks; eles
ficam armazenados pelo próprio FAISS. `.cache/` não é versionado.

A primeira criação pode baixar o tokenizer e gera os embeddings. Mesmo com o
índice salvo, cada consulta ainda usa o modelo de embeddings para a busca e o
modelo de linguagem para produzir a resposta.

## Usar as cadeias

Cada módulo oferece uma função que constrói sua cadeia. A importação não carrega
os PDFs nem executa uma pergunta:

```python
from src.ollama_and_querys.rewrite_chain import build_rewrite_chain

rag_chain = build_rewrite_chain()
answer = rag_chain.invoke("Quais são os benefícios do cartão de crédito?")
print(answer)
```

As demais funções são `build_multi_query_chain`, `build_hyde_chain` e
`build_ollama_chain`, em seus respectivos módulos. Elas também aceitam um
`retriever` já criado para compartilhar o mesmo índice numa execução.

Para executar a pergunta de exemplo de uma cadeia, escolha um dos comandos:

```powershell
python -m src.ollama_and_querys.rewrite_chain
python -m src.ollama_and_querys.multi_query_chain
python -m src.ollama_and_querys.hyde_chain
python -m src.ollama_and_querys.ollama_chain
```

As três primeiras usam `gpt-4o-mini`; a cadeia Ollama usa `gemma3:4b`.

## Gerar perguntas e respostas de referência

O gerador obtém o índice OpenAI compartilhado e recupera os chunks armazenados
nele com `get_indexed_chunks()`. Ele não tem um loader ou splitter próprio. Se
o índice ainda não existir, será criado uma vez antes de gerar as perguntas.

```powershell
python -m src.validations.generate_questions_answers
```

Esse comando usa `QAGenerateChain` com `gpt-4o-mini` e grava
`src/validations/questions_answers.json`, substituindo o dataset anterior. O
formato é mantido:

```json
[
  {
    "qa_pairs": {
      "query": "Pergunta de referência",
      "answer": "Resposta esperada"
    }
  }
]
```

Se o dataset existente já atende à avaliação, basta usá-lo diretamente.

## Avaliar as respostas das cadeias

Para avaliar as primeiras dez perguntas com rewrite:

```powershell
python -m src.validations.validade_questions_answers_rag --chain rewrite --limit 10
```

Troque `--chain` por `multi-query`, `hyde` ou `ollama` para selecionar outra
estratégia. Para comparar as quatro sobre a mesma amostra:

```powershell
python -m src.validations.validade_questions_answers_rag --chain all --limit 10
```

O validador lê as perguntas existentes, executa `chain.invoke(query)` e envia as
respostas ao `QAEvalChain`, que as compara com as respostas de referência. Cada
linha do resultado informa a estratégia, a quantidade avaliada, os acertos e a
acurácia percentual. Em `all`, as três cadeias OpenAI compartilham o mesmo
retriever; Ollama usa seu próprio índice.

O avaliador usa OpenAI `gpt-4o-mini`, inclusive para julgar a cadeia Ollama; por
isso, a avaliação também precisa de `OPENAI_API_KEY`. `--limit` deve ser positivo
e vale 10 por padrão. Se houver menos pares no dataset, todos serão usados.

Para consultar as opções sem executar a avaliação:

```powershell
python -m src.validations.validade_questions_answers_rag --help
```
