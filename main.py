import os
import streamlit as st
import pickle
import time
import gdown
from dotenv import load_dotenv

# Latest correct imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load API Key
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

# Download FAISS if not exists
if not os.path.exists(file_path):
    st.warning("FAISS index not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}", file_path, quiet=False)


# Process Button Logic
if process_clicked:
    all_docs = []

    # Process URLs
    # if urls:
    #     loader = UnstructuredURLLoader(urls=urls)
    #     main_placeholder.text("Loading URL Data...")
    #     data = loader.load()
    #     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #     docs = splitter.split_documents(data)
    #     all_docs.extend(docs)
    if urls:
    loader = AsyncHtmlLoader(urls)
    main_placeholder.text("Loading Data from URLs...")

    html_docs = loader.load()

    # Convert HTML → clean text
    html2text = Html2TextTransformer()
    data = html2text.transform_documents(html_docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    all_docs.extend(docs)


    # Process PDF files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyMuPDFLoader(uploaded_file.name)
            pdf_docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(pdf_docs)
            all_docs.extend(docs)

            os.remove(uploaded_file.name)

    # Build FAISS
    if all_docs:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(all_docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        main_placeholder.text("Vectorstore built successfully! 🎉")


# Query Input
query = st.text_input("Ask a question:")

if query:
    if not os.path.exists(file_path):
        st.error("Vector store not found. Please process data first.")
        st.stop()

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever()

    # Prompt
    template = """
You are an expert research assistant.
Use ONLY the provided context to answer.
If answer is not available, say "I don't know."

Question: {question}

Context:
{context}

Answer:
"""
    prompt = PromptTemplate.from_template(template)

    # Build RAG pipeline (NO deprecated code!)
    rag_chain = (
        RunnableParallel(
            context=retriever, 
            question=RunnablePassthrough()
        )
        | prompt
        | llm
    )

    response = rag_chain.invoke(query)

    st.header("Answer")
    st.write(response)

