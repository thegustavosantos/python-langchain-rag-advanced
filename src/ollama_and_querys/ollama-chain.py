import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.environ.get("LANGSMITH_API_KEY")

FILES = DirectoryLoader("docs/creditcards", glob="*.pdf", loader_cls=PyPDFLoader).load()

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1250,
    chunk_overlap=150
)
chunks = splitter.split_documents(FILES)

embeddings_ollama = OllamaEmbeddings(model="bge-m3:567m")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings_ollama)
retriever = vectorstore.as_retriever()

model = OllamaLLM(model="gemma3:4b")
question = "Quais são os benefícios do cartão de crédito?"

# Resposta sem RAG
# print("Resposta sem RAG: ", model.invoke(question))

# Resposta com RAG
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente os conteúdo fornecido. \n\nContexto:\n{context}"),
        ("human", "{query}")
    ]
)

rag_chain = (
    {
        "context": RunnablePassthrough() | retriever,
        "query": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)
rag_chain.invoke(question)