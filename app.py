try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, RuntimeError):
    # This block runs on your Windows machine
    pass

import streamlit as st
# ... rest of your code ...

import streamlit as st
# ... the rest of your imports follow ...
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os

# 1. Setup the Page
st.set_page_config(page_title="Enterprise AI Data Search", layout="wide")
st.title("📄 Gemini Enterprise RAG System")
st.markdown("### Upload company documents and get instant, grounded answers.")

# 2. Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    if api_key:
    os.environ["os.environ["GOOGLE_API_KEY"] = api_key"] = api_key
    st.info("Get your key at [aistudio.google.com](https://aistudio.google.com/)")

# 3. File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file and api_key:
    # Save the file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing document structure..."):
        # Load and Split the PDF
        loader = PyPDFLoader("temp.pdf")
        data = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        
        # Create Embeddings using Gemini's model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Store in Vector Database (ChromaDB)
        # Using a persist directory makes it more professional
        vectorstore = Chroma.from_documents(docs, embeddings)
        
        # Create the Retrieval Chain - using Gemini 1.5 Flash (fast & cheap)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        # This setup returns the source documents so we can show where the answer came from
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        st.success("Analysis complete! Your AI assistant is ready.")

    # 4. Chat Interface
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": query})
            
            st.markdown("### 🤖 AI Answer:")
            st.write(response["result"])
            
            # Show the Sources (The "Trust" Factor)
            with st.expander("See Sources (Evidence)"):
                for i, doc in enumerate(response["source_documents"][:2]):
                    st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.caption(doc.page_content[:300] + "...")
