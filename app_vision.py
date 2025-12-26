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
