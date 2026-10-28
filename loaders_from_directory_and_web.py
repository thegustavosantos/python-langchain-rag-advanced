import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import MergedDataLoader
from langchain_community.document_loaders import RecursiveUrlLoader



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def build_web_loader():
    URL = "https://forbes.com.br/forbes-tech/2025/08/o-que-explica-o-fracasso-do-chatgpt-5-e-como-a-openai-vai-reagir/"
    return WebBaseLoader(URL)

def build_pdf_directory_loader():
    return DirectoryLoader("docs/creditcards", glob="*.pdf", loader_cls=PyPDFLoader)

def build_recursive_url_loader():
    URL = "https://docs/creditcards.langchain.com/oss/python/langchain/overview"
    return RecursiveUrlLoader(URL)

documents = MergedDataLoader(
    loaders=[build_web_loader(), build_pdf_directory_loader(), build_recursive_url_loader()]
).load()

print(len(documents))