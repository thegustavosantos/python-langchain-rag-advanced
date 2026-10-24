import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModel


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

FILES = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader).load()

# Gerando tokens dos pdfs com huggingface tokenizer
embedding_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
hugging_face_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embedding_tokenizer, chunk_size=512, chunk_overlap=50
)
hugging_face_tokens = hugging_face_splitter.split_documents(FILES)
print(f"Number of documents: {len(FILES)}")
print(f"Number of tokens: {len(hugging_face_tokens)}")


# Quebra das chunks por sentido com LLM da OpenAI
semantic_splitter = SemanticChunker(OpenAIEmbeddings())
semantic_chunks = semantic_splitter.split_documents(FILES)
print(f"Number of semantic chunks with OpenAI: {len(semantic_chunks)}")

# Quebra das chunks por sentido com LLM local OPENSOURCE da HuggingFace
hf_embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
semantic_splitter_hf = SemanticChunker(hf_embedding_model)
semantic_chunks_hf = semantic_splitter_hf.split_documents(FILES)
print(f"Number of semantic chunks with HuggingFace: {len(semantic_chunks_hf)}")