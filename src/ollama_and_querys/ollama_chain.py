from pathlib import Path
import sys

# Permite executar o arquivo diretamente, inclusive fora da raiz do projeto.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

from src.rag.documents import PROJECT_ROOT
from src.rag.vectorstore import get_vectorstore


def build_ollama_chain(retriever: BaseRetriever | None = None) -> Runnable[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    if retriever is None:
        retriever = get_vectorstore("ollama").as_retriever()
    model = OllamaLLM(model="gemma3:4b")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Responda usando exclusivamente os conteúdo fornecido. \n\nContexto:\n{context}"),
            ("human", "{query}"),
        ]
    )

    return (
        {
            "context": RunnablePassthrough() | retriever,
            "query": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


def main() -> None:
    chain = build_ollama_chain()
    print(chain.invoke("Quais são os benefícios do cartão de crédito?"))


if __name__ == "__main__":
    main()
