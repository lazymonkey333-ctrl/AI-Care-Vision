import os
import streamlit as st
from dotenv import load_dotenv
import glob
from typing import List, Any, Union

load_dotenv()

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except ImportError:
    Document = None
    FAISS = None

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Use absolute path relative to this script for data folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_KB_DIR = os.path.join(CURRENT_DIR, "data")

def get_backend_pdfs() -> List[str]:
    """Scan the local data folder for PDF files."""
    if not os.path.exists(BACKEND_KB_DIR):
        os.makedirs(BACKEND_KB_DIR, exist_ok=True)
    return glob.glob(os.path.join(BACKEND_KB_DIR, "*.pdf"))

@st.cache_data(show_spinner="Extracting Medical Knowledge...")
def load_and_split_documents(file_paths: List[str]) -> List[Any]:
    all_docs = []
    if not file_paths: return []
    for fp in file_paths:
        try:
            loader = PyPDFLoader(fp)
            all_docs.extend(loader.load())
        except Exception:
            try:
                import pypdf
                reader = pypdf.PdfReader(fp)
                for i, page in enumerate(reader.pages):
                    all_docs.append(Document(page_content=page.extract_text(), metadata={"source": os.path.basename(fp), "page": i+1}))
            except Exception as e:
                st.error(f"Failed to read {os.path.basename(fp)}: {e}")
    if not all_docs: return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(all_docs)

@st.cache_resource(show_spinner="Building Expert Index...")
def get_vector_store_and_retriever(_splits: List[Any]):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", OPENROUTER_BASE)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=api_key,
        openai_api_base=base_url
    )
    try:
        db = FAISS.from_documents(_splits, embeddings)
        return db.as_retriever(search_kwargs={"k": 4}) # Increase K for more context
    except Exception as e:
        st.error(f"Vector Store Error: {e}")
        return None

def get_retriever(file_paths: List[str] = None):
    targets = file_paths if file_paths else get_backend_pdfs()
    if not targets: return None
    splits = load_and_split_documents(targets)
    if not splits: return None
    return get_vector_store_and_retriever(splits)
