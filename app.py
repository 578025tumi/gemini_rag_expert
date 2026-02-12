import streamlit as st
import os

# 1. STREAMLIT CLOUD FIX (Must be at the very top)
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, RuntimeError):
    pass # This allows the code to run on your local Windows machine

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# PAGE CONFIG & UI
# ==========================================
st.set_page_config(page_title="Gemini RAG Expert", layout="wide")
st.title("📄 Gemini Enterprise Knowledge System")
st.markdown("---")

# ==========================================
# SIDEBAR - CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.markdown("---")
    st.header("📋 Instructions")
    st.write("1. Enter your API Key.")
    st.write("2. Upload a PDF document.")
    st.write("3. Chat with the document logic.")
    
    if st.button("Clear Cache/New Doc"):
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.rerun()

# ==========================================
# CORE RAG LOGIC
# ==========================================
uploaded_file = st.file_uploader("Upload a PDF for AI Analysis", type="pdf")

if uploaded_file and api_key:
    # Set the environment variable for the libraries to use
    os.environ["GOOGLE_API_KEY"] = api_key
    
    #