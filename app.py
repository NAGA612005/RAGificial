import os
import nest_asyncio
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Apply nest_asyncio to allow nested event loops in Flask's threading model
nest_asyncio.apply()

load_dotenv()
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'Uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

rag_chain = None

def create_rag_chain(pdf_path):
    try:
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        if not documents:
            raise ValueError("No content extracted from PDF")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vectordb = FAISS.from_documents(documents, embeddings)
        retriever = vectordb.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0.6)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        return chain
    except Exception as e:
        raise Exception(f"Failed to create RAG chain: {str(e)}")

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
def upload_pdf():
    global rag_chain
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400

        file = request.files['pdf']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Delete previous PDF if exists
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save new file
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Create new RAG chain
        rag_chain = create_rag_chain(path)

        return jsonify({"message": "PDF uploaded successfully"})
    except Exception as e:
        return jsonify({"error": f"Error uploading file: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_chain
    try:
        if rag_chain is None:
            return jsonify({"error": "Upload a PDF first"}), 400
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Question required"}), 400
        question = data.get("question").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        answer = rag_chain.run(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))