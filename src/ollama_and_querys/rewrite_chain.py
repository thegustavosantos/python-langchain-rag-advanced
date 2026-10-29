from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.rag.documents import PROJECT_ROOT
from src.rag.vectorstore import get_vectorstore


def build_rewrite_chain(retriever: BaseRetriever | None = None) -> Runnable[str, str]:
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

    rewriter_prompt = PromptTemplate.from_template(
        """
Gere consulta de pesquisa para o banco de dados de vetores (Vector DB) a partir de uma pergunta do usuário,
permitindo uma resposta mais precisa por meio da busca semantica.
Basta retornar a consulta revisada do Vector DB, entre aspas.

Pergunta do usuário: {user_question}

Consulta revisada do Vector DB:
"""
    )
    rewriter_chain = rewriter_prompt | model_openai | StrOutputParser()

    return (
        {
            "context": RunnablePassthrough() | rewriter_chain | retriever,
            "query": RunnablePassthrough(),
        }
        | prompt
        | model_openai
        | StrOutputParser()
    )


def main() -> None:
    chain = build_rewrite_chain()
    print(chain.invoke("Quais são os benefícios do cartão de crédito?"))


if __name__ == "__main__":
    main()
