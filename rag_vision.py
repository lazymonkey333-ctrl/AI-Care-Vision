import os
import streamlit as st
import openai
import base64
from dotenv import load_dotenv
import rag_vision as _rv

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Care Vision Assistant", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

# --- Header ---
st.title("👁️ AI Care Vision Assistant")

# --- UI Feedback: What docs are actually loaded? ---
pdfs = sorted(_rv.get_backend_pdfs())
pdf_names = [os.path.basename(p) for p in pdfs]

with st.sidebar:
    st.header("1. Expert Archive")
    if pdf_names:
        st.success(f"Backend Sync: {len(pdf_names)} files active.")
        # Show filenames so user knows they are there
        for i, name in enumerate(pdf_names):
            st.caption(f"{i+1}. {name}")
    else:
        st.warning("⚠️ data/ folder is empty! No PDFs found.")
    
    st.markdown("---")
    st.header("2. AI configuration")
    # Dev Mode (Default ON for fast initial testing)
    dev_mode = st.checkbox("Dev Mode (Mock Embeddings)", value=True, help="ON: Instantly load. OFF: Use real API embeddings.")
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1" if dev_mode else "0"
    
    model_name = st.selectbox("Vision Engine", ["openai/gpt-4o-mini", "google/gemini-pro-1.5"], index=0)
    
    if st.button("Force Refresh Knowledge Base"):
        st.session_state.retriever = _rv.get_retriever(pdfs)
        st.success("Archive Refreshed!")

# --- Auto-Loading Logic (Non-blocking) ---
if st.session_state.retriever is None and pdfs:
    status_bar = st.empty()
    status_bar.info("⏳ Authenticating and loading medical archive...")
    try:
        st.session_state.retriever = _rv.get_retriever(pdfs)
        status_bar.success("✅ Archive ready for consultation.")
        import time
        time.sleep(1)
        status_bar.empty()
    except Exception as e:
        status_bar.error(f"Initialization Failed: {e}")

# --- Main Interaction ---
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# A. Image Input
st.subheader("1. Patient Input / Document Photo")
uploaded_image = st.file_uploader("Upload photo (JPG/PNG)", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, caption="Analysis Context", width=350)

st.markdown("---")

# B. Chat Context
st.subheader("2. Medical Consultation")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# C. Logic
if prompt := st.chat_input("Analyze the image or ask about internal guidelines..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting archive..."):
            # 1. RAG Retrieval
            context = ""
            if st.session_state.retriever:
                try:
                    docs = st.session_state.retriever.get_relevant_documents(prompt)
                    context_chunks = []
                    for d in docs:
                        src = d.metadata.get('source', 'Unknown Archive')
                        context_chunks.append(f"DOCUMENT SOURCE: {src}\nCONTENT: {d.page_content}")
                    context = "\n\n".join(context_chunks)
                except Exception: pass
            
            # 2. System Instruction (THE TRAINING)
            training_prompt = (
                "You are an Elite Medical Assistant with access to a specific medical archive. "
                "YOUR ABSOLUTE RULES:\n"
                "1. If INTERNAL ARCHIVE data is provided below, you MUST use it to answer and EXPLICITLY reference the 'DOCUMENT SOURCE' name (e.g., 'According to [Guideline.pdf]...').\n"
                "2. If an image is provided, correlate the visual findings with the text archive data if applicable.\n"
                "3. Provide direct, professional, and data-driven summaries.\n"
                "4. If no relevant info is in the archive, state 'This isn't in my specific archive, but based on general medical knowledge...'"
            )
            if context:
                training_prompt += f"\n\n--- INTERNAL ARCHIVE DATA ---\n{context}"

            # 3. Message Payload
