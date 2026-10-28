import os
from dotenv import load_dotenv

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import CommaSeparatedListOutputParser
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

multi_query_prompt_template = """Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.
Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça estas perguntas alternativas separadas por quebras de linha.
Pergunta original: {question}"""
multi_query_prompt = PromptTemplate.from_template(multi_query_prompt_template)
multi_query_chain = multi_query_prompt | model_openai | CommaSeparatedListOutputParser()

multi_retriever = MultiQueryRetriever(
    retriever=retriever, llm_chain=multi_query_chain
)

multi_query_rag_chain = (
    {
        "context": RunnablePassthrough() | multi_retriever,
        "query": RunnablePassthrough()
    }
    | prompt
    | model_openai
    | StrOutputParser()
)
multi_query_rag_chain.invoke(question)



