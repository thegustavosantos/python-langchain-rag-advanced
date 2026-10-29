import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.evaluation.qa import QAEvalChain
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.ollama_and_querys.hyde_chain import build_hyde_chain
from src.ollama_and_querys.multi_query_chain import build_multi_query_chain
from src.ollama_and_querys.ollama_chain import build_ollama_chain
from src.ollama_and_querys.rewrite_chain import build_rewrite_chain
from src.rag.documents import PROJECT_ROOT
from src.rag.vectorstore import get_vectorstore

QUESTIONS_ANSWERS_FILE = Path(__file__).resolve().parent / "questions_answers.json"


def load_questions_answers(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        try:
            pairs = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"O arquivo {path} não contém um JSON válido.") from error

    if not isinstance(pairs, list) or not pairs:
        raise ValueError("O dataset deve conter uma lista não vazia de pares.")

    questions_answers = []
    for position, pair in enumerate(pairs, start=1):
        qa_pair = pair.get("qa_pairs") if isinstance(pair, dict) else None
        if not isinstance(qa_pair, dict) or any(
            not isinstance(qa_pair.get(key), str) or not qa_pair[key].strip()
            for key in ("query", "answer")
        ):
            raise ValueError(
                f"Par {position} inválido: qa_pairs deve conter query e answer "
                "como textos não vazios."
            )
        questions_answers.append(
            {"query": qa_pair["query"], "answer": qa_pair["answer"]}
        )
    return questions_answers


def generate_answers(
    chain: Runnable[str, str], questions_answers: list[dict[str, str]]
) -> list[dict[str, str]]:
    return [
        {"result": chain.invoke(pair["query"])} for pair in questions_answers
    ]


def validate(
    questions_answers: list[dict[str, str]],
    generations: list[dict[str, str]],
    *,
    evaluator: QAEvalChain,
) -> float:
    if not questions_answers:
        raise ValueError("É necessário pelo menos um par para avaliar a cadeia.")
    if len(questions_answers) != len(generations):
        raise ValueError("A quantidade de respostas deve ser igual à de perguntas.")

    evaluations = evaluator.evaluate(questions_answers, generations)
    correct = 0
    for evaluation in evaluations:
        grade = (
            evaluation["results"].strip().split("\n")[-1].split(":")[-1].strip().upper()
        )
        if grade == "CORRECT":
            correct += 1
    return correct / len(questions_answers)


def main(argv: list[str] | None = None) -> None:
    builders = {
        "rewrite": build_rewrite_chain,
        "multi-query": build_multi_query_chain,
        "hyde": build_hyde_chain,
        "ollama": build_ollama_chain,
    }
    parser = argparse.ArgumentParser(
        description="Avalia as respostas das cadeias RAG com o dataset existente."
    )
    parser.add_argument(
        "--chain",
        choices=[*builders, "all"],
        default="rewrite",
        help="Cadeia a avaliar (padrão: rewrite); all avalia as quatro.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Quantidade máxima de pares a avaliar (padrão: 10).",
    )
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit deve ser um inteiro positivo.")

    try:
        questions_answers = load_questions_answers(QUESTIONS_ANSWERS_FILE)[: args.limit]
    except (OSError, ValueError) as error:
        parser.error(str(error))

    load_dotenv(PROJECT_ROOT / ".env")
    model_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator = QAEvalChain.from_llm(model_openai)
    chain_names = list(builders) if args.chain == "all" else [args.chain]
    retrievers = {}

    for name in chain_names:
        provider = "ollama" if name == "ollama" else "openai"
        if provider not in retrievers:
            retrievers[provider] = get_vectorstore(provider).as_retriever()

        chain = builders[name](retriever=retrievers[provider])
        generations = generate_answers(chain, questions_answers)
        accuracy = validate(questions_answers, generations, evaluator=evaluator)
        correct = round(accuracy * len(questions_answers))
        print(
            f"{name}: {len(questions_answers)} perguntas | "
            f"{correct} acertos | Acurácia: {accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
