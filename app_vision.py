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

        /* UPLOADER STYLING (Miniaturized & Bottom) */
        [data-testid="stFileUploader"] {
            padding-top: 5px;
            margin-top: 10px;
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
    
    # 1. Model Selection (With Free Tier)
    model_name = st.selectbox("Engine", [
        "google/gemini-2.0-flash-exp:free", # Free
        "meta-llama/llama-3.2-11b-vision-instruct:free", # Free
        "openai/gpt-4o-mini", # Paid
        "google/gemini-pro-1.5" # Paid
    ])
    
    # 2. Logic to Force Dev Mode if Free Model is selected
    # This avoids paying for embeddings if the user is broke
    is_free_model = ":free" in model_name
    
    st.markdown("---")
    
    # Dev Mode checkbox - Auto-checked if free model
    if is_free_model:
        st.info("ℹ️ Free Tier active: 'Dev Mode' enabled to save embedding costs.")
        dev_mode = st.checkbox("Dev Mode (Mock Embeddings)", value=True, disabled=False, help="Uses mock search to save money.")
    else:
        dev_mode = st.checkbox("Dev Mode (Mock Embeddings)", value=False)
        
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1" if dev_mode else "0"

    # Files
    if pdfs:
        st.success(f"{len(pdfs)} Archives Connected")
    
    # Debug Info
    st.markdown("---")
    with st.expander("🔍 Debug Info"):
        if "debug_log" in st.session_state:
            st.markdown(st.session_state.debug_log)
        else:
            st.caption("No context.")

if st.session_state.retriever is None and pdfs:
    st.session_state.retriever = _rv.get_retriever(pdfs)

# --- Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image_data" in msg:
            st.image(msg["image_data"], width=250)
        st.markdown(msg["content"])

# --- Uploader (Bottom) ---
uploaded_image = st.file_uploader(
    "📎 Attach Image", 
    type=["jpg", "png", "jpeg"], 
    key=f"uploader_{st.session_state.uploader_key}"
)

# --- Input ---
if prompt := st.chat_input("Message..."):
    # User Msg
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

    # Render AI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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
            
            system_prompt = "You are a helpful medical assistant. Use the archive context if relevant."
            final_prompt = f"{system_prompt}\n\n### ARCHIVE:\n{context}"
            
            payload = [{"type": "text", "text": prompt}]
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
                    max_tokens=1000 # CRITICAL: Strict limit to avoid 402 errors
                )
                ans = res.choices[0].message.content
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
                # Auto-Clear Uploader
                if uploaded_image:
                    st.session_state.uploader_key += 1
                    st.rerun()
                    
            except Exception as e:
                st.error(f"API Error: {e}")
                st.caption("Tip: If you see Error 402, your specific model requires more credits. Try switching to a different ':free' model.")
