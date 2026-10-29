from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "docs" / "creditcards"


@lru_cache(maxsize=1)
def load_chunks() -> list[Document]:
    """Divide os PDFs uma vez e reutiliza os chunks durante esta execução."""
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    documents = DirectoryLoader(
        str(DOCUMENTS_DIR), glob="*.pdf", loader_cls=PyPDFLoader
    ).load()
    if not documents:
        raise ValueError(f"Nenhum documento PDF encontrado em {DOCUMENTS_DIR}.")

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=1250,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)
