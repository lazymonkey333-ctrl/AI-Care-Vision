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

st.title("👁️ AI Care Vision Assistant")

# 获取并显示后台 PDF 列表
pdfs = _rv.get_backend_pdfs()
with st.sidebar:
    st.header("Medical Archive")
    if pdfs:
        st.success(f"{len(pdfs)} files detected.")
        for p in pdfs: st.caption(f"- {os.path.basename(p)}")
    
    st.markdown("---")
    dev_mode = st.checkbox("Dev Mode", value=True)
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
        if st.session_state.retriever:
            docs = st.session_state.retriever.get_relevant_documents(prompt)
            context = "\n\n".join([f"[Source: {d.metadata.get('source')}] {d.page_content}" for d in docs])
        
        payload = [{"type": "text", "text": f"Context from documents:\n{context}\n\nQuestion: {prompt}"}]
        if uploaded_image:
            b64 = base64.b64encode(uploaded_image.read()).decode('utf-8')
            payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"))
        res = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": payload}],
            extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"}
        )
        ans = res.choices[0].message.content
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
