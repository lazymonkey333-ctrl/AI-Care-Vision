import os
import streamlit as st
import openai
import base64
from dotenv import load_dotenv
import rag_vision as _rv

load_dotenv()
st.set_page_config(
    page_title="AI Care Vision", 
    page_icon="🧡",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- THEME: WARM CARE (Ultra-Safe Version) ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif;
        }
        
        .stApp {
            background-color: #FDFCF8;
        }
        
        [data-testid="stSidebar"] {
            background-color: #F6F3E6;
        }
        
        /* Chat Bubbles */
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
             background-color: #FFFFFF;
             border: 1px solid #EFEBE0;
             border-radius: 10px;
        }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
             background-color: #FFF0E3;
             border: 1px solid #FFE0C2;
             border-radius: 10px;
        }
        
        /* Headers */
        h1, h2, h3, p {
            color: #4A3B32;
        }
        
        /* Button */
        div.stButton > button {
            background-color: #FFB74D;
            color: white;
            border: none;
        }

        /* Smaller Uploader Text */
        [data-testid="stFileUploader"] label {
            font-size: 0.8rem;
            color: #888;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

# --- Helper: Image Encoding ---
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# --- Personas ---
PERSONAS = {
    "🛡️ Standard Expert": "You are an Elite Medical Assistant. Use internal archive data if possible.",
    "💕 Empathetic Caregiver": "You are a warm, compassionate healthcare companion. Use simple language.",
    "🔬 Strict Analyst": "You are a rigorous data analyst. Be concise and data-driven.",
    "👴 Elderly Friendly": "Speak slowly and clearly. Use metaphors. Focus on safety."
}

st.title("🧡 AI Care Assistant")

# --- Backend Logic ---
pdfs = _rv.get_backend_pdfs()
with st.sidebar:
    st.header("🧠 Settings")
    selected_persona_name = st.selectbox("Persona", list(PERSONAS.keys()), index=0)
    current_system_prompt = PERSONAS[selected_persona_name]
    
    st.markdown("---")
    if pdfs:
        st.success(f"{len(pdfs)} Archives Active")
        with st.expander("File List"):
            for p in pdfs: st.caption(f"- {os.path.basename(p)}")
    else:
        st.error("No PDFs found")
        
    st.markdown("---")
    dev_mode = st.checkbox("Dev Mode", value=False)
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1" if dev_mode else "0"
    model_name = st.selectbox("Engine", ["openai/gpt-4o-mini", "google/gemini-pro-1.5"])

if st.session_state.retriever is None and pdfs:
    with st.spinner("Loading..."):
        st.session_state.retriever = _rv.get_retriever(pdfs)

# --- Chat History Loop ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Display Image if present in history
        if "image_data" in msg:
            st.image(msg["image_data"], width=300)
        st.markdown(msg["content"])

# --- Input Area (Linear Layout) ---
st.markdown("---")
uploaded_image = st.file_uploader("📎 Upload photo (optional)", type=["jpg", "png", "jpeg"], label_visibility="visible")

if prompt := st.chat_input("Ask a medical question..."):
    # 1. Prepare User Message Object
    user_msg_obj = {"role": "user", "content": prompt}
    
    # 2. Handle Image
    base64_img = None
    if uploaded_image:
        # Save exact bytes for display history
        user_msg_obj["image_data"] = uploaded_image.getvalue() 
        base64_img = encode_image(uploaded_image)
        
    st.session_state.messages.append(user_msg_obj)
    
    # Render immediately
    with st.chat_message("user"):
        if uploaded_image: st.image(uploaded_image, width=300)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = ""
        context_debug = "No context."
        if st.session_state.retriever:
            try:
                docs = st.session_state.retriever.get_relevant_documents(prompt)
                if docs:
                    context = "\n\n".join([f"[Source: {d.metadata.get('source')}] {d.page_content}" for d in docs])
                    context_debug = "\n\n".join([f"**📄 {d.metadata.get('source')}**\n> {d.page_content[:200]}..." for d in docs])
            except Exception: pass
        
        with st.expander("🔍 Debug Context"): st.markdown(context_debug)
        
        final_prompt = f"{current_system_prompt}\n\n### ARCHIVE:\n{context}"
        
        payload = [{"type": "text", "text": prompt}]
        if base64_img:
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"))
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": payload}],
                extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"}
            )
            ans = res.choices[0].message.content
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"Error: {e}")
