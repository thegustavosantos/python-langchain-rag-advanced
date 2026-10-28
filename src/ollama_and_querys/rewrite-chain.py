import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.environ.get("LANGSMITH_API_KEY")

FILES = DirectoryLoader("docs/creditcards", glob="*.pdf", loader_cls=PyPDFLoader).load()
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1250,
    chunk_overlap=150
)
chunks = splitter.split_documents(FILES)

embeddings_openai = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=api_key)
vectorstore = FAISS.from_documents(chunks, embedding=embeddings_openai)
retriever = vectorstore.as_retriever()

model_openai = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)

question = "Quais são os benefícios do cartão de crédito?"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente os conteúdo fornecido. \n\nContexto:\n{context}"),
        ("human", "{query}")
    ]
)

rewriter_prompt_template = """
Gere consulta de pesquisa para o banco de dados de vetores (Vector DB) a partir de uma pergunta do usuário,
permitindo uma resposta mais precisa por meio da busca semantica.
Basta retornar a consulta revisada do Vector DB, entre aspas.

Pergunta do usuário: {user_question}

Consulta revisada do Vector DB:
"""
rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)
rewriter_chain = rewriter_prompt | model_openai | StrOutputParser()

rewriter_rag_chain = (
    {
        "context": RunnablePassthrough() | rewriter_chain | retriever,
        "query": RunnablePassthrough()
    }
    | prompt
    | model_openai
    | StrOutputParser()
)
rewriter_rag_chain.invoke(question)

