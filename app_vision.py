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
        /* This moves the uploader visually closer to the input */
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
        [data-testid="stFileUploader"] button {
             /* Hide the "Browse files" button if you want it even smaller, 
                but keeping it for usability */
             padding: 0px 5px;
             font-size: 10px;
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

# --- PERSONAS RE-INJECTED ---
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
    
    # 1. Persona Selector
    selected_persona_name = st.selectbox("Assistant Style", list(PERSONAS.keys()), index=0)
    current_system_prompt = PERSONAS[selected_persona_name]
    
    st.markdown("---")
    
    # 2. Model Selector (Strictly Working Free Models)
    # Removing broken Llama model to prevent 404
    model_name = st.selectbox("Engine (Free Tier)", [
        "google/gemini-2.0-flash-exp:free", # Primary
        "google/gemini-pro-1.5", # Backup (Might be paid/limited)
    ])
    
    # 3. Auto-DevMode Logic
    is_free_model = ":free" in model_name
    if is_free_model:
        st.info("ℹ️ Free Tier: 'Dev Mode' enabled by default.")
        dev_mode = st.checkbox("Dev Mode", value=True, help="Mock embeddings for free users")
    else:
        dev_mode = st.checkbox("Dev Mode", value=False)
        
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1" if dev_mode else "0"

    # Files
    st.markdown("---")
    if pdfs:
        st.success(f"{len(pdfs)} Archives Connected")
    
    # Debug Info
    with st.expander("🔍 Debug Info"):
        if "debug_log" in st.session_state:
            st.markdown(st.session_state.debug_log)
        else:
            st.caption("No context.")

if st.session_state.retriever is None and pdfs:
    st.session_state.retriever = _rv.get_retriever(pdfs)

# --- MAIN LAYOUT LOGIC ---
# 1. Chat Container (Top)
chat_container = st.container()

# 2. Uploader Container (Bottom, just above input)
# We place this physically AFTER the chat container in the code, 
# so it renders below the messages.
uploader_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Only display image if this specific message HAS one
            if "image_data" in msg:
                st.image(msg["image_data"], width=250)
            st.markdown(msg["content"])

with uploader_container:
    # Use key to allow programmatic clearing
    uploaded_image = st.file_uploader(
        "📎 Attach Image (Optional)", 
        type=["jpg", "png", "jpeg"], 
        key=f"uploader_{st.session_state.uploader_key}"
    )

# --- Input ---
if prompt := st.chat_input("Message..."):
    # User Msg
    user_msg_obj = {"role": "user", "content": prompt}
    base64_img = None
    
    # Handle Image ONLY if currently present
    if uploaded_image:
        user_msg_obj["image_data"] = uploaded_image.getvalue()
        base64_img = encode_image(uploaded_image)
    
    st.session_state.messages.append(user_msg_obj)
    
    # Render User Immediately
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
            
            # Combine PERSONA + ARCHIVE CONTEXT
            final_prompt = f"{current_system_prompt}\n\n### ARCHIVE:\n{context}"
            
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
                    max_tokens=1000 # Strict Limit
                )
                ans = res.choices[0].message.content
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
                # CRITICAL: Auto-Clear Uploader Logic
                # If we used an image, increment key to FORCE reset of uploader widget
                if uploaded_image:
                    st.session_state.uploader_key += 1
                    st.rerun()
                    
            except Exception as e:
                st.error(f"API Error: {e}")
