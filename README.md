# 📄 RAGificial – Ask Questions to Your File (Powered by Gemini AI)

Live Demo 👉 [https://ragificial-1.onrender.com/](https://ragificial-1.onrender.com/)

RAGificial is a web app that lets you upload a file and ask natural language questions about its content. It uses **Google Gemini AI** and **LangChain** to read, embed, retrieve, and answer based on the uploaded document.

---

## 🚀 Features

- 🔍 Upload any File (PDF, CSV, XLSX) and ask questions from it
- 🧠 Uses **Gemini 2.0 Flash** model for intelligent answers
- 🔗 Retrieval-Augmented Generation (RAG) using **LangChain**
- ⚡ Fast document search using **FAISS**
- 🌐 Clean, responsive UI for both desktop and mobile
- 🔐 API key is securely loaded from `.env`

---

## 🖼️ Live Application

👉 **Try it here:**  
🌐 [https://ragificial-1.onrender.com/](https://ragificial-1.onrender.com/)

---

## 🧰 Tech Stack

- **Backend**: Python, Flask
- **AI**: Gemini AI (via LangChain)
- **Embedding**: GoogleGenerativeAIEmbeddings
- **Vector DB**: FAISS
- **PDF Parsing**: PDFPlumber
- **CSV, XLSX Parsing**: Pandas
- **Frontend**: HTML, CSS (responsive)
- **Deployment**: Render.com

---

## 📦 Installation

```bash
git clone https://github.com/NAGA612005/ragificial.git
cd ragificial
pip install -r requirements.txt

