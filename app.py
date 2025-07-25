import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import pdfplumber
import pandas as pd

load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
rag_chain = None

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text

    elif ext in ['.csv', '.tsv']:
        df = pd.read_csv(file_path)
        return df.to_string()

    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        return df.to_string()

    else:
        return "[Unsupported file type]"

def create_rag_chain(file_path):
    content = extract_text_from_file(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([content])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectordb = FAISS.from_documents(texts, embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0.6)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    global rag_chain
    try:
        # Clear old files
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        file = request.files['pdf']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)

            rag_chain = create_rag_chain(file_path)
            return jsonify({"message": f"{file.filename} uploaded and processed."})
        return jsonify({"error": "No file uploaded"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_chain
    if rag_chain is None:
        return jsonify({"error": "Upload a file first."})
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question required"})
    answer = rag_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
