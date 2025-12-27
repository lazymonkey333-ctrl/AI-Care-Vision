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

# --- THEME: WARM CARE (Polished) ---
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

        /* -------------------------------------------
           UPLOADER STYLING (Miniaturized) 
           ------------------------------------------- */
        [data-testid="stFileUploader"] {
            padding-top: 0px;
            margin-top: 20px;
        }
        [data-testid="stFileUploader"] section {
            padding: 10px !important;
            background-color: #fafafa;
            border: 1px dashed #ddd;
        }
        /* Hide giant icon if possible or shrink elements */
        [data-testid="stFileUploader"] div, 
        [data-testid="stFileUploader"] span, 
        [data-testid="stFileUploader"] small {
            font-size: 12px !important; /* Tiny Text */
            line-height: 1.2 !important;
        }
        /* Minimize the dropzone height */
        [data-testid="stFileUploader"] button {
             display: none; /* Hide the Browse button to keep it clean if D&D works, or just make small */
        }
        
        /* Headers */
        h1, h2, h3, p {
            color: #4A3B32;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None
# Uploader Key for Auto-Reset
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

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

# --- Title ---
st.title("🧡 AI Care Assistant")

# --- Backend Logic ---
pdfs = _rv.get_backend_pdfs()
with st.sidebar:
    st.header("🧠 Personalization")
    selected_persona_name = st.selectbox("Style", list(PERSONAS.keys()), index=0)
    current_system_prompt = PERSONAS[selected_persona_name]
    
    st.markdown("---")
    if pdfs:
        st.success(f"{len(pdfs)} Archives Connected")
    
    # DEBUG CONTEXT (Moved to Sidebar to hide it)
    # The user wanted it "small icon" or "to the side". Sidebar is perfect.
    st.markdown("---")
    with st.expander("🔍 Debug Info"):
        if "debug_log" in st.session_state:
            st.markdown(st.session_state.debug_log)
        else:
            st.caption("No context loaded yet.")
            
    dev_mode = st.checkbox("Dev Mode", value=False)
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1" if dev_mode else "0"
    model_name = st.selectbox("Engine", ["openai/gpt-4o-mini", "google/gemini-pro-1.5"])

if st.session_state.retriever is None and pdfs:
    st.session_state.retriever = _rv.get_retriever(pdfs)

# --- 1. Chat History (First) ---
# This ensures messages are at the top, uploader pushes down
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image_data" in msg:
            st.image(msg["image_data"], width=250)
        st.markdown(msg["content"])

# --- 2. Uploader (Middle/Bottom) ---
# Key trick: Render uploader AFTER history so it sits at the bottom of the scroll flow
# Using a key allows us to reset it programmatically
uploaded_image = st.file_uploader(
    "📎 Attach Image (Single Turn Analysis)", 
    type=["jpg", "png", "jpeg"], 
    key=f"uploader_{st.session_state.uploader_key}"
)

# --- 3. Input (Fixed Bottom) ---
if prompt := st.chat_input("Ask about your health..."):
    # A. User Msg
    user_msg_obj = {"role": "user", "content": prompt}
    base64_img = None
    
    # B. Handle Image (Only if present THIS turn)
    if uploaded_image:
        user_msg_obj["image_data"] = uploaded_image.getvalue()
        base64_img = encode_image(uploaded_image)
    
    st.session_state.messages.append(user_msg_obj)
    
    # Force Rerun to display user message immediately? 
    # Streamlit runs top-to-bottom. We need to render the new user message NOW.
    with st.chat_message("user"):
        if uploaded_image: st.image(uploaded_image, width=250)
        st.markdown(prompt)

    # C. AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # RAG
            context = ""
            debug_text = "No context."
            if st.session_state.retriever:
                try:
                    docs = st.session_state.retriever.get_relevant_documents(prompt)
                    if docs:
                        context = "\n\n".join([f"[Source: {d.metadata.get('source')}] {d.page_content}" for d in docs])
                        debug_text = "\n\n".join([f"**📄 {d.metadata.get('source')}**\n> {d.page_content[:200]}..." for d in docs])
                except Exception: pass
            
            # Save Debug Info to Session State for Sidebar
            st.session_state.debug_log = debug_text
            
            # Prompt Construction
            final_prompt = f"{current_system_prompt}\n\n### ARCHIVE:\n{context}"
            payload = [{"type": "text", "text": prompt}]
            
            # Only append image if it was uploaded THIS specific time
            if base64_img:
                payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

            try:
                client = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"), 
                    base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
                )
                res = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": payload}],
                    extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"},
                    max_tokens=2048 # Prevent 402 errors by limiting reservation
                )
                ans = res.choices[0].message.content
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
                # D. Auto-Clear Uploader Logic
                # Increase key -> next render creates a fresh uploader
                if uploaded_image:
                    st.session_state.uploader_key += 1
                    st.rerun() # Refresh to clear the file input visual
                    
            except Exception as e:
                st.error(f"Error: {e}")
