from flask import Flask, render_template, jsonify, request
from src.helper import hugging_face_embedding
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
import os
import logging

app = Flask(__name__)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
try:
    logger.info("Initializing application components...")
    
    # Pinecone initialization
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key not found in environment variables")
        
    index_name = "medical-chatbot"

    logger.info(f"Connecting to Pinecone with index: {index_name}")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Initialize embedding
    logger.info("Initializing embedding model...")
    embedding = hugging_face_embedding()
    
    # Initialize vector store
    logger.info("Initializing Pinecone vector store...")
    doc_search = PineconeLangChain.from_existing_index(index_name, embedding)
    
    # Create prompt template
    logger.info("Creating prompt template...")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}

    # Load LLM model
    logger.info("Loading LLM model...")
    model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        config={'temperature': 0.7}
    )

    # Create QA chain
    logger.info("Creating QA chain...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_search.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        # Get message from both form and JSON requests
        if request.is_json:
            msg = request.json.get("msg")
        else:
            msg = request.form.get("msg")
            
        if not msg:
            return jsonify({
                "status": "error",
                "message": "Empty message"
            }), 400
            
        logger.info(f"Received query: {msg}")
        
        result = qa.invoke({"query": msg})
        response = result.get("result", "No response generated")
        
        logger.info(f"Generated response: {response}")
        return jsonify({
            "status": "success",
            "message": response
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)