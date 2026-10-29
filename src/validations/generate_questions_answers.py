import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.evaluation.qa import QAGenerateChain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.rag.documents import PROJECT_ROOT
from src.rag.vectorstore import get_indexed_chunks, get_vectorstore

OUTPUT_FILE = Path(__file__).resolve().parent / "questions_answers.json"


def generate_questions_answers(chunks: list[Document]) -> list[dict]:
    """Gera os pares de perguntas e respostas a partir de chunks existentes."""
    if not chunks:
        raise ValueError("É necessário pelo menos um chunk para gerar perguntas.")

    load_dotenv(PROJECT_ROOT / ".env")
    model_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = QAGenerateChain.from_llm(model_openai)
    return qa_chain.apply([{"doc": chunk.page_content} for chunk in chunks])


def main() -> None:
    vectorstore = get_vectorstore("openai")
    chunks = get_indexed_chunks(vectorstore)
    questions_answers = generate_questions_answers(chunks)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(questions_answers, file, ensure_ascii=False, indent=2)

    print(f"Perguntas e respostas salvas em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
