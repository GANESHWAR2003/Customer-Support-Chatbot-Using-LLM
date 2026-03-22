from flask import Flask, request, render_template, jsonify, redirect, url_for
import fitz  # PyMuPDF
from dotenv import load_dotenv
import os
import uuid

# LangChain (latest)
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load env
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Models
model = ChatGroq(api_key=groq_api_key, model="llama-3.1-70b-versatile", temperature=0)
parser = StrOutputParser()
llm = model | parser

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Storage
pdf_text_storage = {}
chat_history = {}
current_conversation_id = None

# ================= PDF =================
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf = fitz.open(pdf_path)
    for page in pdf:
        text += page.get_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")

# ================= AI CHAIN =================
def get_chain():
    prompt = PromptTemplate(
        template="""
You are an AI assistant.

Context:
{context}

Question:
{question}

Answer clearly:
""",
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=google_api_key
    )

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_files', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/upload_files', methods=['POST'])
def upload_files():
    global current_conversation_id

    files = request.files.getlist('files')
    file_info = []

    for file in files:
        if file and file.filename.endswith('.pdf'):
            file_id = str(uuid.uuid4())
            path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.pdf")
            file.save(path)

            text = extract_text_from_pdf(path)
            pdf_text_storage[file_id] = text

            file_info.append({"id": file_id, "name": file.filename})

    # create vector db
    combined = "\n\n".join(pdf_text_storage.values())
    chunks = get_text_chunks(combined)
    get_vector_store(chunks)

    current_conversation_id = str(uuid.uuid4())
    chat_history[current_conversation_id] = []

    return jsonify({"message": "Uploaded successfully", "files": file_info})

@app.route('/query')
def query():
    if not current_conversation_id:
        return redirect(url_for('home'))
    return render_template('query.html')

@app.route('/ask', methods=['POST'])
def ask():
    global current_conversation_id

    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question"}), 400

    if not pdf_text_storage:
        return jsonify({"error": "Upload PDF first"}), 400

    # Load vector DB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    context = "\n".join([doc.page_content for doc in docs])

    chain = get_chain()

    # ✅ FIXED (LCEL)
    response = chain.invoke({
        "context": context,
        "question": question
    })

    chat_history[current_conversation_id].append({
        "question": question,
        "response": response
    })

    return jsonify({"response": response})

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)