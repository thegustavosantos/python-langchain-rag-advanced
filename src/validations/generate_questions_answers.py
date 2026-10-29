import json
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_classic.evaluation.qa import QAGenerateChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
lang_key = os.environ.get("LANGSMITH_API_KEY")

FILES = DirectoryLoader("docs/creditcards", glob="*.pdf", loader_cls=PyPDFLoader).load()
LLM_MODEL_NAME = "gpt-4o-mini"

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1250,
    chunk_overlap=150
)
chunks = splitter.split_documents(FILES)


model_openai = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)
qa_chain = QAGenerateChain.from_llm(model_openai)

questions_answers = qa_chain.apply(
    [ {"doc": chunk.page_content } for chunk in chunks ]
)

OUTPUT_FILE = Path(__file__).resolve().parent / "questions_answers.json"
with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(questions_answers, file, ensure_ascii=False, indent=2)

print(f"Perguntas e respostas salvas em: {OUTPUT_FILE}")
