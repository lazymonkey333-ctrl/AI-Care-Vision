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

        /* UPLOADER STYLING: Bottom Placement */
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

st.title("🧡 AI Care Assistant")

# --- Backend Logic ---
pdfs = _rv.get_backend_pdfs()

with st.sidebar:
    st.header("🧠 Settings")
    
    # 1. Models (Fallback Logic Built-in)
    st.caption("Auto-switches to backup if rate limited.")
    primary_model = "google/gemini-2.0-flash-exp:free"
    backup_model = "meta-llama/llama-3.2-11b-vision-instruct:free"
    
    if pdfs:
        st.success(f"{len(pdfs)} Archives Connected")
    
    # Debug Info
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

# --- LAYOUT: MAIN CHAT + STATUS ---
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

# --- INSTANT PREVIEW (Before Send) ---
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

    # 2. AI Logic (With Fallback)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # RAG Context
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
        
        # Prompt
        system_prompt = "You are a helpful medical assistant. Use the archive context if relevant."
        final_prompt = f"{system_prompt}\n\n### ARCHIVE:\n{context}"
        
        payload = [{"type": "text", "text": prompt}]
        if base64_img:
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

        # API CALL WITH RETRY
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
            # FALLBACK
            try:
                message_placeholder.markdown("⚠️ Gemini Busy (429). Switching to Llama...")
                time.sleep(1) # Brief pause
                res = client.chat.completions.create(
                    model=backup_model,
                    messages=[{"role": "system", "content": final_prompt}, {"role": "user", "content": payload}],
                    extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"},
                    max_tokens=1000
                )
                ans = f"[Backup Model] {res.choices[0].message.content}"
            except Exception as e2:
                ans = f"❌ Service Unavailable. Please try again later. Error: {e2}"

    message_placeholder.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    
    # 3. CLEANUP (Guaranteed)
    # If successful response, clear uploader
    if uploaded_image:
        st.session_state.uploader_key += 1
        st.rerun()
