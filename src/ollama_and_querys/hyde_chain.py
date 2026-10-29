from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.rag.documents import PROJECT_ROOT
from src.rag.vectorstore import get_vectorstore


def build_hyde_chain(retriever: BaseRetriever | None = None) -> Runnable[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    if retriever is None:
        retriever = get_vectorstore("openai").as_retriever()
    model_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Responda usando exclusivamente os conteúdo fornecido. \n\nContexto:\n{context}"),
            ("human", "{query}"),
        ]
    )

    hyde_prompt = PromptTemplate.from_template(
        """
Responda a pergunta da melhor forma possivel em no minimo 2 paragrafos.
Pergunta: {user_question}
Responda:
"""
    )
    hyde_chain = hyde_prompt | model_openai | StrOutputParser()

    return (
        {
            "context": RunnablePassthrough() | hyde_chain | retriever,
            "query": RunnablePassthrough(),
        }
        | prompt
        | model_openai
        | StrOutputParser()
    )


def main() -> None:
    chain = build_hyde_chain()
    print(chain.invoke("Quais são os benefícios do cartão de crédito?"))


if __name__ == "__main__":
    main()
