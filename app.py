try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, RuntimeError):
    pass

import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup
st.set_page_config(page_title="Gemini RAG Expert", layout="wide")
st.title("📄 Gemini Enterprise RAG System")

# 2. Sidebar - API Key and Session Management
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if st.button("Clear Chat History"):
        st.session_state.vector_store = None
        st.rerun()

# 3. Initialize "Brain" (Session State)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 4. File Upload Logic
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    os.environ["AIzaSyCnBvIfLUvROwOCHaNsIb-qfLaBHv25dNw"] = api_key
    
    # Only analyze if we haven't already analyzed this session
    if st.session_state.vector_store is None:
        with st.status("Analyzing document... (This may take 10-20 seconds)") as status:
            try:
                # Save PDF locally
                st.write("Reading file...")
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and Split
                st.write("Splitting text into chunks...")
                loader = PyPDFLoader("temp.pdf")
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(data)
                
                # Embeddings and Vector Store
                st.write("Creating vector database...")
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                
                # Create the store and save to session state
                vectorstore = Chroma.from_documents(docs, embeddings)
                st.session_state.vector_store = vectorstore
                
                status.update(label="Analysis Complete!", state="complete")
                st.success("You can now ask questions!")
            
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                status.update(label="Analysis Failed", state="error")

# 5. Question & Answer Logic
if st.session_state.vector_store:
    query = st.chat_input("Ask a question about your document")
    
    if query:
        # Show user message
        with st.chat_message("user"):
            st.write(query)
            
        # Generate Answer
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            
            prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the provided context.
            Context: {context}
            Question: {input}
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching document..."):
                    response = retrieval_chain.invoke({"input": query})
                    st.write(response["answer"])
                    
                    # Optional: Show sources
                    with st.expander("Show relevant sections"):
                        for doc in response["context"]:
                            st.write(f"---\n{doc.page_content[:300]}...")
                            
        except Exception as e:
            st.error(f"Error generating answer: {e}")