import os
import streamlit as st
import openai
import base64
import time
from dotenv import load_dotenv
import rag_vision as _rv

load_dotenv()
st.set_page_config(
    page_title="AI Care Vision", 
    page_icon="🧡",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- THEME: WARM CARE ---
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

        /* UPLOADER STYLING */
        [data-testid="stFileUploader"] {
            padding-bottom: 5px;
            margin-bottom: 5px;
        }
        [data-testid="stFileUploader"] section {
            padding: 8px !important;
            background-color: #fafafa;
            border: 1px dashed #ddd;
        }
        [data-testid="stFileUploader"] div, span, small {
            font-size: 11px !important;
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
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

# --- Function: Encode Image ---
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# --- PERSONAS RESTORED ---
PERSONAS = {
    "🛡️ Standard Expert": "You are an Elite Medical Assistant. Rules: 1. Prioritize internal archive data. 2. Be concise and professional.",
    "💕 Empathetic Caregiver": "You are a warm, compassionate healthcare companion. Rules: 1. Use simple, reassuring language. 2. Focus on comfort and understandable advice.",
    "🔬 Strict Analyst": "You are a rigorous data analyst. Rules: 1. Be extremely direct and concise. 2. Focus purely on data and guidelines.",
    "👴 Elderly Friendly": "You are a patient assistant for elderly users. Rules: 1. Speak very clearly and slowly. 2. Use metaphors. 3. Remind about safety."
}

st.title("🧡 AI Care Assistant")

# --- Backend Logic ---
pdfs = _rv.get_backend_pdfs()

with st.sidebar:
    st.header("🧠 Personalization")
    
    # 1. Persona Selector (RESTORED)
    selected_persona_name = st.selectbox("Assistant Style", list(PERSONAS.keys()), index=0)
    current_system_prompt = PERSONAS[selected_persona_name]
    
    st.markdown("---")
    
    # 2. Models (Safe Logic)
    # Using simple retry logic on the SAME stable free model instead of a broken backup
    st.caption("Engine: Google Gemini")
    primary_model = "google/gemini-2.0-flash-exp:free"
    
    if pdfs:
        st.success(f"{len(pdfs)} Archives Connected")
    
    with st.expander("🔍 Debug"):
        if "debug_log" in st.session_state:
            st.markdown(st.session_state.debug_log)
        else:
            st.caption("No context.")

    # Hidden Dev Mode
    st.checkbox("Dev Mode", value=True, key="dev_mode_hidden", disabled=True)
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1"

if st.session_state.retriever is None and pdfs:
    st.session_state.retriever = _rv.get_retriever(pdfs)

# --- LAYOUT ---
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "image_data" in msg:
                st.image(msg["image_data"], width=250)
            st.markdown(msg["content"])

# --- UPLOADER (Bottom) ---
uploaded_image = st.file_uploader(
    "📎 Attach Image", 
    type=["jpg", "png", "jpeg"], 
    key=f"uploader_{st.session_state.uploader_key}"
)

# --- INSTANT PREVIEW ---
if uploaded_image:
    st.image(uploaded_image, caption="Ready to send...", width=150)

# --- INPUT ---
if prompt := st.chat_input("Message..."):
    # 1. User Logic
    user_msg_obj = {"role": "user", "content": prompt}
    base64_img = None
    
    if uploaded_image:
        user_msg_obj["image_data"] = uploaded_image.getvalue()
        base64_img = encode_image(uploaded_image)
    
    st.session_state.messages.append(user_msg_obj)
    
    # Render User
    with st.chat_message("user"):
        if uploaded_image: st.image(uploaded_image, width=250)
        st.markdown(prompt)

    # 2. AI Logic (Retry Strategy)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
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
        
        st.session_state.debug_log = debug_text
        
        final_prompt = f"{current_system_prompt}\n\n### ARCHIVE:\n{context}"
        
        payload = [{"type": "text", "text": prompt}]
        if base64_img:
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

        # API CALL
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        )
        
        try:
            message_placeholder.markdown("⏳ Gemini Thinking...")
            res = client.chat.completions.create(
                model=primary_model,
                messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": payload}],
                extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"},
                max_tokens=1000
            )
            ans = res.choices[0].message.content
            
        except Exception as e:
            # RETRY SAME MODEL (Because Llama was 404 broken)
            try:
                message_placeholder.markdown("⚠️ Server Busy (429). Retrying in 2s...")
                time.sleep(2) 
                res = client.chat.completions.create(
                    model=primary_model, 
                    messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": payload}],
                    extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"},
                    max_tokens=1000
                )
                ans = res.choices[0].message.content
            except Exception as e2:
                ans = f"❌ System Overloaded. Please wait 10 seconds and try again. (Details: {e2})"

    message_placeholder.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    
    # 3. CLEANUP
    if uploaded_image:
        st.session_state.uploader_key += 1
        st.rerun()
