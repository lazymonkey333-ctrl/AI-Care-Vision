import os
import streamlit as st
import openai
import base64
from dotenv import load_dotenv
import rag_vision as _rv

load_dotenv()

# 1. 强制收起侧边栏，页面设为英文
st.set_page_config(page_title="AI Care Vision", layout="wide", initial_sidebar_state="collapsed")

if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

st.title("👁️ AI Care Vision Assistant")

# 2. 将上传区域移至主页，确保一眼就能看到
st.subheader("1. Upload Medical Document")
uploaded_image = st.file_uploader("Upload a photo of a prescription or report", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, caption="Analysis Subject", width=300)

st.markdown("---")
st.subheader("2. Chat with Expert")

# 打印历史记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 侧边栏仅作为后台设置
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", ["openai/gpt-4o-mini", "google/gemini-pro-1.5"])
    if st.button("Reload Knowledge Base"):
        st.session_state.retriever = _rv.get_retriever()
        st.success("Indexing Done")

# 输入框逻辑
if prompt := st.chat_input("Ask about the image or medical guidelines..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            context = ""
            if st.session_state.retriever:
                docs = st.session_state.retriever.get_relevant_documents(prompt)
                context = "\n".join([d.page_content for d in docs])
            
            # 构造多模态消息负载
            content_payload = [{"type": "text", "text": prompt}]
            if uploaded_image:
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(uploaded_image)}"}
                })

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"))
            
            # 使用正确的 extra_headers 修复报错
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"You are a medical assistant. Guidelines:\n{context}"},
                    {"role": "user", "content": content_payload}
                ],
                extra_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "AI Care Vision"}
            )
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
