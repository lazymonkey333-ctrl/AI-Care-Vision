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

# --- THEME: WARM CARE (Safe CSS Version) ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* 1. Global Font */
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600&display=swap');
        
        html, body, .stApp {
            font-family: 'Nunito', sans-serif !important;
            background-color: #FDFCF8 !important; /* Creamy White */
        }
        
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #F6F3E6 !important; /* Warm Beige */
            border-right: 1px solid #EADDCD;
        }

        /* 2. Chat Bubbles */
        /* User Bubble (Right) */
        .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
             background-color: #FFF0E3; /* Soft Peach */
             border: 1px solid #FFE0C2;
             border-radius: 12px;
        }
        
        /* Assistant Bubble (Left) - Use generic selector to be safe */
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
             background-color: #FFFFFF;
             border: 1px solid #EFEBE0;
             border-radius: 12px;
        }
        
        /* 3. Headers & Accents */
        h1, h2, h3, .stSubheader {
            color: #5D4037 !important; /* Warm Brown */
            font-weight: 600 !important;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #FFB74D !important; /* Warm Orange */
            color: white !important;
            border-radius: 20px !important;
            border: none !important;
        }
        
        /* 4. Input Box Styling (Ensure Visibility) */
        .stChatInputContainer {
            padding-bottom: 20px;
        }
        .stChatInput textarea {
            background-color: #FFFFFF !important;
            color: #333333 !important;
            border: 1px solid #E0E0E0 !important;
        }

        /* REMOVED AGGRESSIVE HIDING RULES TO FIX UI BUGS */
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

# --- Pre-defined Personas ---
PERSONAS = {
    "🛡️ Standard Expert": (
        "You are an Elite Medical Assistant. Rules:\n"
        "1. Prioritize internal medical archive data if available.\n"
        "2. Be concise, professional, and empathetic.\n"
        "3. If the archive lacks info, use general medical knowledge but add a disclaimer."
    ),
    "💕 Empathetic Caregiver": (
        "You are a warm, compassionate healthcare companion. Rules:\n"
        "1. Use simple, reassuring language suitable for patients or elderly users.\n"
        "2. Avoid complex jargon. Explain medical terms simply.\n"
        "3. Focus on comfort and actionable care advice based on the archive."
    ),
    "🔬 Strict Analyst": (
        "You are a rigorous medical data analyst. Rules:\n"
        "1. Be extremely direct and concise. No fluff or pleasantries.\n"
        "2. Focus purely on data, statistics, and clinical guidelines from the archive.\n"
        "3. If data is missing, state 'Insufficient Data' immediately."
    ),
    "👴 Elderly Friendly": (
        "You are a patient assistant for elderly users. Rules:\n"
        "1. Speak very clearly and slowly (conceptually).\n"
        "2. Use metaphors to explain conditions.\n"
        "3. Remind them gently about safety and dosage."
    )
}

st.title("🧡 AI Care Assistant")
st.caption("Your warm, personal healthcare companion.")

# --- Sidebar ---
pdfs = _rv.get_backend_pdfs()
with st.sidebar:
    st.header("🧠 Personality")
    selected_persona_name = st.selectbox("Choose Style", list(PERSONAS.keys()), index=0)
    current_system_prompt = PERSONAS[selected_persona_name]
    
    st.markdown("---")
    st.header("📂 Archive")
    if pdfs:
        st.success(f"{len(pdfs)} files active.")
        with st.expander("Show Files"):
            for p in pdfs: st.caption(f"- {os.path.basename(p)}")
    else:
        st.error("No PDFs in data/ folder")
    
    st.markdown("---")
    dev_mode = st.checkbox("Dev Mode", value=False)
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1" if dev_mode else "0"
    model_name = st.selectbox("Engine", ["openai/gpt-4o-mini", "google/gemini-pro-1.5"])

if st.session_state.retriever is None and pdfs:
    with st.spinner("Warming up knowledge base..."):
        st.session_state.retriever = _rv.get_retriever(pdfs)

# --- Layout: Columns for Upload & Chat ---
# Use a unified layout to avoid elements getting hidden
col1, col2 = st.columns([1, 2])

with col1:
    st.info("� **Upload Image**")
    uploaded_image = st.file_uploader("Select medical photo...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_image:
        st.image(uploaded_image, caption="Analysis Context", use_column_width=True)

with col2:
    st.info("💬 **Consultation History**")
    # Chat Container
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("*No messages yet. Start the conversation below!*")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- Logic ---
# Chat input is naturally fixed at bottom, should be visible now CSS is fixed
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

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
        
        with st.expander("🔍 Debug Context"):
            st.markdown(context_debug)
        
        final_prompt = current_system_prompt
        if context:
            final_prompt += f"\n\n### ARCHIVE DATA:\n{context}"
        
        payload = [{"type": "text", "text": prompt}]
        if uploaded_image:
            b64 = base64.b64encode(uploaded_image.read()).decode('utf-8')
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

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
