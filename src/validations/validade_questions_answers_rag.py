import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.environ.get("LANGSMITH_API_KEY")

LLM_MODEL_NAME = "gpt-4o-mini"

model_openai = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)
eval_chain = QAEvalChain.from_llm(model_openai)

def validate(questions_answers, generations):
    # questions_answers: query, answer
    # generations: result
    response = eval_chain.evaluate(questions_answers, generations)
    correct = 0
    for i, e in enumerate(questions_answers):
        correct = corretas + (1 if response[i]["results"].split("\n")[-1].split(":")[-1].strip() == "CORRECT" else 0)
    return correct/len(questions_answers)

QUESTIONS_ANSWERS_FILE = Path(__file__).resolve().parent / "questions_answers.json"
with QUESTIONS_ANSWERS_FILE.open(encoding="utf-8") as file:
    pairs = json.load(file)

questions_answers = [p["qa_pairs"] for p in pairs]
print(len(questions_answers))
print(questions_answers[0])

result = []
for pr in questions_answers[:10]
    questions_answers_result.append({"result" : rag_chain})