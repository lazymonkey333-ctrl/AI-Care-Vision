import os
import streamlit as st
import openai
import base64
from dotenv import load_dotenv
import rag_vision as _rv

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AI Care Vision Pro", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

# --- Auto-Initialize Knowledge Base ---
# If retriever is None and PDFs exist, load them automatically
pdfs = _rv.get_backend_pdfs()
if st.session_state.retriever is None and pdfs:
    with st.spinner("Expert Knowledge Loading... (One-time setup)"):
        st.session_state.retriever = _rv.get_retriever(pdfs)

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# --- Header ---
st.title("👁️ AI Care Vision Assistant")
st.caption("Auto-loading backend knowledge base from 'data/' folder.")

# --- UI: Image Upload (Prominent) ---
st.subheader("1. Medical Document Analysis")
uploaded_image = st.file_uploader("Upload prescription, report, or symptom photo", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, caption="Current Analysis Subject", width=350)

st.markdown("---")
st.subheader("2. Consultation")

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- Sidebar (Silent Backend Settings) ---
with st.sidebar:
    st.header("Expert Settings")
    model_name = st.selectbox("Vision Engine", ["openai/gpt-4o-mini", "google/gemini-pro-1.5"], index=0)
    if pdfs:
        st.success(f"Backend Sync: {len(pdfs)} documents active.")
    if st.button("Force Re-index Knowledge Base"):
        st.session_state.retriever = _rv.get_retriever()
        st.success("Indexing Updated")

# --- AI Logic ---
if prompt := st.chat_input("Analyze the image above or ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting internal medical archive..."):
            # 1. Retrieval
            context = ""
            if st.session_state.retriever:
                try:
                    docs = st.session_state.retriever.get_relevant_documents(prompt)
                    # Include source metadata to prove it's reading the PDFs
                    context_parts = []
                    for d in docs:
                        source_name = d.metadata.get('source', 'Unknown Document')
                        content_parts.append(f"[Source: {source_name}] {d.page_content}")
                    context = "\n\n".join(content_parts)
                except Exception:
                    pass
            
            # 2. "Training" through advanced System Persona
            system_prompt = (
                "You are an Elite Medical Assistant. Your goal is to provide highly targeted, accurate advice. "
                "CRITICAL RULES:\n"
                "1. If internal medical context is provided, you MUST explicitly reference it in your answer (e.g., 'According to the guideline...').\n"
                "2. If an image is provided, analyze its medical implications with high precision.\n"
                "3. Be direct and avoid generic disclaimers unless strictly necessary.\n"
                "4. Always maintain a professional, reassuring, and data-driven tone."
            )
            if context:
                system_prompt += f"\n\nUSE THESE INTERNAL GUIDELINES:\n{context}"

            content_payload = [{"type": "text", "text": prompt}]
            if uploaded_image:
                content_payload.append({
                    "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(uploaded_image)}"}
                })

            # API Call (with memory)
            try:
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"))
                api_msgs = [{"role": "system", "content": system_prompt}]
                api_msgs.extend(st.session_state.messages[-10:])
                
                # Replace the last user message with multi-modal content
                api_msgs[-1] = {"role": "user", "content": content_payload}

                response = client.chat.completions.create(
                    model=model_name,
                    messages=api_msgs,
                    extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision Pro"}
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
