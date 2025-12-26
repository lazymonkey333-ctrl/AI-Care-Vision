import os
import streamlit as st
import openai
import base64
from dotenv import load_dotenv
import rag_vision as _rv

load_dotenv()
st.set_page_config(page_title="AI Care Vision", layout="wide", initial_sidebar_state="collapsed")

if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

# --- Pre-defined Personas (The "Brain" Configurations) ---
PERSONAS = {
    "🛡️ Standard Expert (Default)": (
        "You are an Elite Medical Assistant. Rules:\n"
        "1. Prioritize internal medical archive data if available.\n"
        "2. Be concise, professional, and empathetic.\n"
        "3. If the archive lacks info, use general medical knowledge but add a disclaimer."
    ),
    "💕 Empathetic Caregiver (For Patients)": (
        "You are a warm, compassionate healthcare companion. Rules:\n"
        "1. Use simple, reassuring language suitable for patients or elderly users.\n"
        "2. Avoid complex jargon. Explain medical terms simply.\n"
        "3. Focus on comfort and actionable care advice based on the archive."
    ),
    "🔬 Strict Analyst (For Researchers)": (
        "You are a rigorous medical data analyst. Rules:\n"
        "1. Be extremely direct and concise. No fluff or pleasantries.\n"
        "2. Focus purely on data, statistics, and clinical guidelines from the archive.\n"
        "3. If data is missing, state 'Insufficient Data' immediately."
    ),
    "👴 Elderly Friendly (Simplified)": (
        "You are a patient assistant for elderly users. Rules:\n"
        "1. Speak very clearly and slowly (conceptually).\n"
        "2. Use metaphors to explain conditions.\n"
        "3. Remind them gently about safety and dosage."
    )
}

st.title("👁️ AI Care Vision Assistant")

# 获取并显示后台 PDF 列表
pdfs = _rv.get_backend_pdfs()
with st.sidebar:
    st.header("🧠 AI Personality Mode")
    
    # 1. Persona Selector (Safe for users)
    selected_persona_name = st.selectbox(
        "Choose Assistant Style", 
        list(PERSONAS.keys()),
        index=0
    )
    current_system_prompt = PERSONAS[selected_persona_name]
    
    st.info(f"**Current Role:** {selected_persona_name}")

    st.markdown("---")
    st.header("📂 Medical Archive")
    if pdfs:
        st.success(f"{len(pdfs)} files active.")
        with st.expander("View File List"):
            for p in pdfs: st.caption(f"- {os.path.basename(p)}")
    else:
        st.error("No PDFs found in data/ folder!")
    
    st.markdown("---")
    dev_mode = st.checkbox("Dev Mode (Mock Embeddings)", value=False)
    os.environ["RAG_USE_RANDOM_EMBEDDINGS"] = "1" if dev_mode else "0"
    model_name = st.selectbox("Vision Engine", ["openai/gpt-4o-mini", "google/gemini-pro-1.5"])

if st.session_state.retriever is None and pdfs:
    with st.spinner("Initializing Knowledge..."):
        st.session_state.retriever = _rv.get_retriever(pdfs)

st.subheader("1. Upload Image")
uploaded_image = st.file_uploader("Upload photo", type=["jpg", "png", "jpeg"])
if uploaded_image: st.image(uploaded_image, width=300)

st.markdown("---")
st.subheader("2. Consultation")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the document or image..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        context = ""
        context_debug = "No context retrieved."
        
        if st.session_state.retriever:
            try:
                docs = st.session_state.retriever.get_relevant_documents(prompt)
                if docs:
                    context = "\n\n".join([f"[Source: {d.metadata.get('source')}] {d.page_content}" for d in docs])
                    context_debug = "\n\n".join([f"**📄 Source: {d.metadata.get('source')}**\n> {d.page_content[:200]}..." for d in docs])
                else:
                    context_debug = "⚠️ Search returned 0 results."
            except Exception as e:
                context_debug = f"❌ Search Error: {e}"
        
        with st.expander("🔍 Debug: View Retrieved Context"):
            st.markdown(context_debug)
        
        # Combine Selected Persona + Context
        final_system_prompt = current_system_prompt
        if context:
            final_system_prompt += f"\n\n### INTERNAL ARCHIVE CONTEXT:\n{context}\n\n(Use the above context to answer the user's question)"
        
        payload = [{"type": "text", "text": prompt}]
        if uploaded_image:
            b64 = base64.b64encode(uploaded_image.read()).decode('utf-8')
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"))
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": payload}
                ],
                extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"}
            )
            ans = res.choices[0].message.content
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"API Error: {e}")
