import os
import streamlit as st
import pickle
import time
import gdown
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader, PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API key
groq_api_key = st.secrets["GROQ_API_KEY"]

# Title
st.title("RAG Research Tool 📘")

# Sidebar inputs
st.sidebar.header("Input Data")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", accept_multiple_files=True, type=["pdf"]
)

process_clicked = st.sidebar.button("Process Data")

FAISS_FILE = "faiss_store.pkl"

main_placeholder = st.empty()

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=groq_api_key
)

# ----------------------------- PROCESSING -----------------------------
if process_clicked:
    all_docs = []

    # Process URLs
    if urls:
        main_placeholder.text("Fetching and processing URLs...")
        try:
            loader = UnstructuredURLLoader(urls=urls, continue_on_failure=True)
            data = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = splitter.split_documents(data)
            all_docs.extend(docs)

        except Exception as e:
            st.error("Error loading URLs:")
            st.write(str(e))

    # Process PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                loader = PyMuPDFLoader(uploaded_file.name)
                pdf_docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                docs = splitter.split_documents(pdf_docs)
                all_docs.extend(docs)

            finally:
                os.remove(uploaded_file.name)

    # Build FAISS
    if all_docs:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(all_docs, embeddings)

            with open(FAISS_FILE, "wb") as f:
                pickle.dump(vectorstore, f)

            st.success("Vectorstore created successfully! 🎉")

        except Exception as e:
            st.error("Vectorstore creation failed:")
            st.write(str(e))


# ----------------------------- QUERY -----------------------------
query = st.text_input("Ask a question:")

if query:
    if not os.path.exists(FAISS_FILE):
        st.error("Please process data first. No FAISS index found.")
        st.stop()

    # Load vector store
    with open(FAISS_FILE, "rb") as f:
        vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Retrieve docs
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        st.error("Error retrieving documents:")
        st.write(str(e))
        st.stop()

    # Build context
    if docs:
        context = "\n\n".join([d.page_content[:500] for d in docs])
    else:
        context = ""

    # Prompt template
    prompt = f"""
You are a precise research assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Question: {query}

Context:
{context}

Answer:
"""

    # Ask LLM
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        st.error("LLM invocation failed:")
        st.write(str(e))
        st.stop()

    st.subheader("Answer")
    st.write(str(response))
