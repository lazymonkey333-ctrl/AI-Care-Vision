import os
import streamlit as st
import openai
import base64
from dotenv import load_dotenv
import rag_vision as _rv
from datetime import datetime

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Care Vision", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- Helper: Image Encoding ---
def encode_image(image_file):
    """Encode an uploaded image to base64."""
    return base64.b64encode(image_file.read()).decode('utf-8')

# --- Header ---
st.title("👁️ AI Care Vision Assistant")
st.caption("A multi-modal platform that can see your medical documents and analyze them based on internal knowledge.")

# --- Sidebar: Configuration & Image Upload ---
with st.sidebar:
    st.header("1. Document Analysis")
    uploaded_image = st.file_uploader("Upload Medical Image (Prescription, Report, etc.)", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Document", use_column_width=True)

    st.markdown("---")
    st.header("2. Knowledge Base Status")
    pdfs = _rv.get_backend_pdfs()
    if pdfs:
        st.success(f"Ready: {len(pdfs)} PDFs in backend.")
        if st.button("Initialize Knowledge Index"):
            with st.spinner("Indexing..."):
                st.session_state.retriever = _rv.get_retriever()
                st.success("Indexing Complete!")
    else:
        st.warning("Please upload PDFs to the 'data/' folder.")

    st.markdown("---")
    st.header("3. Model Settings")
    # Using OpenRouter as the provider
    model_name = st.selectbox("Select Vision Model", ["openai/gpt-4o-mini", "google/gemini-flash-1.5", "anthropic/claude-3-haiku"])
    st.info("Ensure your OpenRouter API Key is set in the environment or secrets.")

# --- Chat Interface ---
# Scrollable messaging area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Explain this document or ask about medical guidelines..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # 1. Context Retrieval (Text-based)
            context = ""
            if st.session_state.retriever:
                try:
                    docs = st.session_state.retriever.get_relevant_documents(prompt)
                    context = "\n".join([d.page_content for d in docs])
                except Exception as e:
                    st.warning(f"RAG Error: {e}")

            # 2. Construct Multi-modal Payload
            system_prompt = "You are a professional medical assistant. Analyze the user's input. "
            if context:
                system_prompt += f"Consider the following internal guidelines:\n{context}"
            
            # Message structure for OpenRouter/OpenAI Vision
            content_list = [{"type": "text", "text": prompt}]
            if uploaded_image:
                base64_image = encode_image(uploaded_image)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            # Prepare API Message (with memory)
            # We only send text memory for simplicity, but the current turn contains the image
            api_messages = [{"role": "system", "content": system_prompt}]
            # Memory - text only for previous rounds
            for m in st.session_state.messages[-6:-1]:
                api_messages.append({"role": m["role"], "content": m["content"]})
            
            # Current turn - multi-modal
            api_messages.append({"role": "user", "content": content_list})

            # 3. Call API via OpenRouter
            try:
                client = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
                )
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=api_messages,
                    headers={
                        "HTTP-Referer": "https://ai-care-platform.streamlit.app", # Replace with your site URL
                        "X-Title": "AI Care Vision", # Optional name
                    }
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Vision API Error: {e}")
