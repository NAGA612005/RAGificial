import os
import nest_asyncio
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Apply nest_asyncio to allow nested event loops in Flask's threading model
nest_asyncio.apply()
import pdfplumber
import pandas as pd

load_dotenv()
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'Uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

rag_chain = None

def create_rag_chain(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'

    elif ext in ['.csv']:
        df = pd.read_csv(file_path)
        text = df.to_string()

    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        text = df.to_string()

    else:
        raise ValueError("Unsupported file type")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectordb = FAISS.from_documents(texts, embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0.6)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain

@app.route('/')
def home():
    global rag_chain
    try:
        # Clear the Uploads folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # Reset rag_chain
        rag_chain = None
        return render_template("index.html")
    except Exception as e:
        return jsonify({"error": f"Error clearing uploads: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    global rag_chain
    return jsonify({"has_pdf": rag_chain is not None})

@app.route('/upload', methods=['POST'])
def upload_file():
    global rag_chain

    # Delete previous PDF if exists
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save new file
    file = request.files['file']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)

        # Create new RAG chain
        rag_chain = create_rag_chain(path)

        return jsonify({"message": "File uploaded successfully."})
    return jsonify({"error": "No file uploaded"}), 400


@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_chain
    if rag_chain is None:
        return jsonify({"error": "Upload a File first."})
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question required"})
    answer = rag_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

