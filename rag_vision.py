import os
import streamlit as st
from dotenv import load_dotenv
import glob
from typing import List, Any, Union

# Load environment variables
load_dotenv()

# LangChain imports
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
except ImportError:
    # Fallback placeholders if libs are missing
    Document = None
    FAISS = None

# Configuration
# For Vision platform, we default to OpenAI-compatible base for OpenRouter
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
BACKEND_KB_DIR = "data"

def get_backend_pdfs() -> List[str]:
    """Scan the data folder for PDF files."""
    if not os.path.exists(BACKEND_KB_DIR):
        os.makedirs(BACKEND_KB_DIR, exist_ok=True)
    return glob.glob(os.path.join(BACKEND_KB_DIR, "*.pdf"))

@st.cache_data(show_spinner="Loading Vision Knowledge Base...")
def load_and_split_documents(file_paths: List[str]) -> List[Any]:
    all_docs = []
    if not file_paths:
        return []
        
    for fp in file_paths:
        try:
            loader = PyPDFLoader(fp)
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error reading {fp}: {e}")
    
    if not all_docs:
        return []
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(all_docs)

@st.cache_resource(show_spinner="Indexing documents...")
def get_vector_store_and_retriever(_splits: List[Any]):
    # Use OpenRouter for embeddings if possible, or fallback
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", OPENROUTER_BASE)
    
    # We use a standard embedding model supported by OpenRouter
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=api_key,
        openai_api_base=base_url
    )
    
    try:
        db = FAISS.from_documents(_splits, embeddings)
        return db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Vector Store Error: {e}")
        return None

def get_retriever(file_paths: List[str] = None):
    targets = file_paths if file_paths else get_backend_pdfs()
    if not targets:
        return None
    splits = load_and_split_documents(targets)
    if not splits:
        return None
    return get_vector_store_and_retriever(splits)
