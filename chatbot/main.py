import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from flask import Flask, render_template, request, jsonify, session
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For session management

# Load the PDF and extract text
pdf_path = "Deep Learning_ Recurrent Neural Networks in Python_ LSTM, GRU, and more RNN machine learning architectures in Python and Theano (Machine Learning in Python) ( PDFDrive ).pdf"  # Update with your PDF path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Convert text chunks into embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embedding_model)

# Save the vector DB for offline use
vector_db.save_local("faiss_index")

# Load LLM model (DeepSeek/Gemma)
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"  # Change to Gemma if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)

def generate_answer(query):
    retriever = vector_db.as_retriever()
    retriever_chain = RetrievalQA.from_chain_type(llm=HuggingFacePipeline(model=model, tokenizer=tokenizer), retriever=retriever)
    return retriever_chain.run(query)

@app.route("/")
def home():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    response = generate_answer(user_query)
    
    # Store conversation history
    session["chat_history"].append({"user": user_query, "bot": response})
    session.modified = True
    
    return jsonify({"response": response, "chat_history": session["chat_history"]})

if __name__ == "__main__":
    app.run(debug=True)
