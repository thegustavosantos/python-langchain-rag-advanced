from typing import Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.rag.documents import PROJECT_ROOT, load_chunks

INDEX_DIR = PROJECT_ROOT / ".cache" / "rag"


def get_vectorstore(provider: Literal["openai", "ollama"] = "openai") -> FAISS:
    """Carrega o índice do provedor ou cria e salva no primeiro uso."""
    load_dotenv(PROJECT_ROOT / ".env")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        index_path = INDEX_DIR / "openai-text-embedding-3-small"
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model="bge-m3:567m")
        index_path = INDEX_DIR / "ollama-bge-m3-567m"
    else:
        raise ValueError(f"Provedor de embeddings desconhecido: {provider}.")

    if index_path.exists():
        # Carregamos apenas os arquivos criados pelo próprio projeto.
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vectorstore = FAISS.from_documents(load_chunks(), embedding=embeddings)
    vectorstore.save_local(str(index_path))
    return vectorstore


def get_indexed_chunks(vectorstore: FAISS) -> list[Document]:
    """Recupera os chunks já armazenados no FAISS, sem dividir os PDFs."""
    chunks = []
    for position in sorted(vectorstore.index_to_docstore_id):
        document_id = vectorstore.index_to_docstore_id[position]
        document = vectorstore.docstore.search(document_id)
        if not isinstance(document, Document):
            raise ValueError(
                f"Índice FAISS inválido: documento {document_id} não encontrado."
            )
        chunks.append(document)
    return chunks
