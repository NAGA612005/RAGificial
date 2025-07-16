import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
rag_chain = None

def create_rag_chain(pdf_path):
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
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

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global rag_chain

    # Delete previous PDF if exists
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save new file
    file = request.files['pdf']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(path)

        # Create new RAG chain
        rag_chain = create_rag_chain(path)

        return jsonify({"message": "PDF uploaded successfully."})
    return jsonify({"error": "No file uploaded"}), 400


@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_chain
    if rag_chain is None:
        return jsonify({"error": "Upload a PDF first."})
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question required"})
    answer = rag_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

