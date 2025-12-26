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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_KB_DIR = os.path.join(CURRENT_DIR, "data")

def get_backend_pdfs() -> List[str]:
    if not os.path.exists(BACKEND_KB_DIR): os.makedirs(BACKEND_KB_DIR, exist_ok=True)
    return sorted(glob.glob(os.path.join(BACKEND_KB_DIR, "*.pdf")))

@st.cache_data(show_spinner="Reading archive...")
def load_and_split_documents(file_paths: List[str]):
    all_docs = []
    if not file_paths: return []
    for fp in file_paths:
        try:
            loader = PyPDFLoader(fp)
            pages = loader.load()
            for p in pages: p.metadata["source"] = os.path.basename(fp)
            all_docs.extend(pages)
        except Exception as e:
            st.error(f"Error reading {os.path.basename(fp)}: {e}")
    if not all_docs: return []
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)

@st.cache_resource(show_spinner="Building index...")
def get_vector_store_and_retriever(_splits):
    is_dev = os.getenv("RAG_USE_RANDOM_EMBEDDINGS") == "1"
    if is_dev:
        class Mock:
            def __init__(self, d): self.d = d
            def get_relevant_documents(self, q): return self.d[:5]
        return Mock(_splits)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"), openai_api_base=os.getenv("OPENAI_API_BASE", OPENROUTER_BASE))
    return FAISS.from_documents(_splits, embeddings).as_retriever(search_kwargs={"k": 5})

def get_retriever(file_paths=None):
    targets = file_paths if file_paths else get_backend_pdfs()
    if not targets: return None
    splits = load_and_split_documents(targets)
    return get_vector_store_and_retriever(splits) if splits else None
