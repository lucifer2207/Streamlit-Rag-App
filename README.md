# 📚 AI-Powered PDF & URL Research Tool

### 🔍 Ask Questions & Summarize Documents with LLM-Powered Retrieval-Augmented Generation (RAG)

## 🚀 Overview
This project is an **AI-powered research assistant** that allows users to upload **PDFs** or enter **URLs**, process the content into a **vector database (FAISS)**, and interact with the documents using **natural language queries**. The app is built using **LangChain, FAISS, Hugging Face Embeddings, and Streamlit**, with **persistent storage in Google Drive** for efficient retrieval over time.

## 🎯 Features
✅ **Upload PDFs & URLs** → Extracts and processes content automatically.  
✅ **Ask Questions from Documents** → Uses **FAISS Vector Search** + LLMs for retrieval.  
✅ **Persistent Storage with FAISS** → Avoids re-processing documents every session.  
✅ **Deployed on Streamlit Cloud** → Easy access, no local setup required.  
 

## 🛠 Tech Stack
- **Programming Language**: Python 🐍
- **Frameworks & Libraries**: LangChain, FAISS, Hugging Face, PyMuPDF, gdown
- **LLM Provider**: Groq Cloud (Llama 3.3-70B Versatile)
- **Frontend**: Streamlit
- **Storage**: FAISS (Google Drive backup for `.pkl` file)
- **Deployment**: Streamlit Cloud

## ⚙️ How It Works
1️⃣ **Upload a PDF or Provide a URL** → The app extracts text from the document.  
2️⃣ **Content is Chunked & Embedded** → Uses Hugging Face embeddings & FAISS indexing.  
3️⃣ **Persistent Storage** → Saves the FAISS index to Google Drive for future queries.  
4️⃣ **Ask Questions from Your Document** → Retrieves relevant sections & answers using LLMs.   

## 🏗 Setup & Installation
### 🔹 Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Set Up API Keys (For Local Testing)
Create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "your_actual_api_key_here"
```

### 🔹 Run the Application
```bash
streamlit run main.py
```

## 🌐 Deployment on Streamlit Cloud
This app is deployed on **Streamlit Cloud**. To deploy your own version:
1. Push your project to **GitHub**.
2. Go to **[Streamlit Cloud](https://share.streamlit.io/)** → Create a new app.
3. Select your **GitHub repository**.
4. **Manually add API keys** under `Manage App > Secrets`.
5. Click **Deploy** 🚀.

## 📌 Example Use Cases
🔹 **Legal & Compliance** → Search policies, contracts, legal documents.  
🔹 **Academic Research** → Ask questions from research papers & PDFs.  
🔹 **Business & Finance** → Analyze reports, summaries, and knowledge retrieval.  

## 📝 Future Improvements
🔹 **Support for More File Types** (DOCX, TXT, CSV)  
🔹 **Integration with Vector DBs like Pinecone, ChromaDB**  
🔹 **Real-time Streaming Data Processing**  

## 🤝 Contribution
Feel free to **fork the repo** and submit a pull request! For major changes, please open an issue first to discuss your ideas.  

## 📩 Contact
For any questions or suggestions, reach out via **[LinkedIn]([https://www.linkedin.com/in/your-profile](https://www.linkedin.com/in/alok-yadav-0a71a4246/)** or **open an issue** in this repository.

---

### ⭐ If you like this project, don't forget to give it a star! ⭐

