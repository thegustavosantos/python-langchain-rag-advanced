from dotenv import load_dotenv
from langchain_classic.retrievers.multi_query import (
    LineListOutputParser,
    MultiQueryRetriever,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.rag.documents import PROJECT_ROOT
from src.rag.vectorstore import get_vectorstore


def build_multi_query_chain(
    retriever: BaseRetriever | None = None,
) -> Runnable[str, str]:
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

    multi_query_prompt = PromptTemplate.from_template(
        """Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.
Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça estas perguntas alternativas separadas por quebras de linha.
Pergunta original: {question}"""
    )
    multi_query_chain = multi_query_prompt | model_openai | LineListOutputParser()
    multi_retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=multi_query_chain
    )

    return (
        {
            "context": RunnablePassthrough() | multi_retriever,
            "query": RunnablePassthrough(),
        }
        | prompt
        | model_openai
        | StrOutputParser()
    )


def main() -> None:
    chain = build_multi_query_chain()
    print(chain.invoke("Quais são os benefícios do cartão de crédito?"))


if __name__ == "__main__":
    main()
