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
        try:
            with st.spinner("Step 1: Reading PDF..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("temp.pdf")
                data = loader.load()

            with st.spinner("Step 2: Chunks & Embeddings..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(data)
                
                # We specify the model explicitly
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                
                # We use a unique 'persist_directory' for Windows stability
                vectorstore = Chroma.from_documents(
                    documents=docs, 
                    embedding=embeddings,
                    persist_directory="./chroma_db" 
                )
                st.session_state.vector_store = vectorstore
                st.success("Analysis Complete!")
                
        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            # This will print the full error to your terminal so you can tell me what it says!
            print(f"DEBUG ERROR: {e}")
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