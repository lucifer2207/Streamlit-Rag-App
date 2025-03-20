import os
import streamlit as st
import pickle
import time
import gdown
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyMuPDFLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

  # Load environment variables (Groq API key)

groq_api_key = st.secrets["GROQ_API_KEY"]
if not groq_api_key:
    st.error("Groq API key not found. Please check your .env file.")
    st.stop()

st.title("Question - Summary - Research Tool 📈")
st.sidebar.title("Input Data")

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

# Collect PDFs
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

process_clicked = st.sidebar.button("Process Data")
file_path = "faiss_store_groq.pkl"
gdrive_file_id = "1g77MsT-99yvYyCJAUsDYKBhsyY6PfaN8"  # Replace with your actual Google Drive file ID

main_placeholder = st.empty()
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.9, api_key=groq_api_key)

# Check if FAISS index exists; if not, download from Google Drive
if not os.path.exists(file_path):
    st.warning("FAISS index not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, file_path, quiet=False)

if process_clicked:
    all_docs = []
    
    # Process URLs
    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Loading Data from URLs... ✅✅✅")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
    
    # Process PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyMuPDFLoader(uploaded_file.name)
            main_placeholder.text(f"Processing PDF: {uploaded_file.name} ✅✅✅")
            pdf_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(pdf_docs)
            all_docs.extend(docs)
            
            os.remove(uploaded_file.name)  # Clean up after processing
    
    if all_docs:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_groq = FAISS.from_documents(all_docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building... ✅✅✅")
        time.sleep(2)
        
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_groq, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            st.header("Answer")
            st.write(result["answer"])
            
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)