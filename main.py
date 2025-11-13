import os
import streamlit as st
import pickle
import time
import gdown
from dotenv import load_dotenv

# LangChain updated imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

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
gdrive_file_id = "1g77MsT-99yvYyCJAUsDYKBhsyY6PfaN8"

main_placeholder = st.empty()

# LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=groq_api_key
)

# Download FAISS Index if not exists
if not os.path.exists(file_path):
    st.warning("FAISS index not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, file_path, quiet=False)

# Process Data Button
if process_clicked:
    all_docs = []

    # Process URL data
    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Loading Data from URLs... ⏳")
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = splitter.split_documents(data)
        all_docs.extend(docs)

    # Process PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyMuPDFLoader(uploaded_file.name)
            main_placeholder.text(f"Processing PDF: {uploaded_file.name} ⏳")
            pdf_docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(pdf_docs)
            all_docs.extend(docs)

            os.remove(uploaded_file.name)

    # Build FAISS Index
    if all_docs:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(all_docs, embeddings)

        main_placeholder.text("Building Embedding Vectorstore... ⏳")
        time.sleep(1)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

# Query Section
query = main_placeholder.text_input("Ask a Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

        result = chain.invoke({"query": query})

        st.header("Answer")
        st.write(result["result"])

        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.write(doc.metadata)
            st.write(doc.page_content[:200] + "...")
