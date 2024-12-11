from flask import Flask, request, jsonify
import os
import shutil
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from huggingface_hub import login
import time
from langchain_community.document_loaders import PyPDFLoader
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

LOGIN_TOKEN = os.getenv("HUGGINGFACE_LOGIN_TOKEN")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chromaDB")  

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./upload")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
EMBEDDINGS = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

LLM = HuggingFacePipeline.from_model_id(
    model_id="gpt2-medium",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.1, "max_new_tokens": 100, "do_sample": True}
)

def split_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return text_splitter.split_documents(docs_before_split)

def load_file_into_db(file_path):
    docs_after_split = split_documents(file_path)
    
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
        db.add_documents(docs_after_split)
        return db
    else:
        return Chroma.from_documents(docs_after_split, EMBEDDINGS, persist_directory=CHROMA_PATH)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Chunk and load the file into chroma db
        load_file_into_db(file_path)
        
        return jsonify({"message": "File uploaded and processed successfully"}), 200

def initialize_retrieval_qa():
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3, 'fetch_k': 3})
    PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with five sentences maximum.
    3. Only answer once, do not repeat the answer.

    {context}

    Question: {question}

    Helpful Answer:
    """
    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )

retrievalQA = initialize_retrieval_qa()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('query')
    if not question:
        return jsonify({"error": "Query parameter is required"}), 400
    start = time.time()
    result = retrievalQA.invoke({"query": question})
    end = time.time()
    relevant_docs = result['source_documents']
    relevant_docs_info = []

    for i, doc in enumerate(relevant_docs):
       
        formatted_content = " ".join(doc.page_content.split()).replace("\n", " ")

        doc_info = {
            "document_number": i + 1,
            "source_file": doc.metadata.get('source', 'Unknown'),
            "page": doc.metadata.get('page', 'Unknown'),
            "content": formatted_content
        }
        relevant_docs_info.append(doc_info)

    return jsonify({
        "query": result["query"],
        "result": result["result"],
        "execution_time": end - start,
        "relevant_documents": relevant_docs_info,
        "total_relevant_documents": len(relevant_docs)
    })

