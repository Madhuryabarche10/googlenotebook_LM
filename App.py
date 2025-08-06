import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Get the directory of the current script (app.py)
# This ensures load_dotenv looks for .env relative to app.py's location
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(BASE_DIR, '.env')

# Load environment variables from .env file
load_dotenv(dotenv_path=dotenv_path) # <--- Pass the explicit path here

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Configuration from Environment Variables ---
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
# Choose ONE based on your LLM provider:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not LLAMAPARSE_API_KEY:
    raise ValueError("LLAMAPARSE_API_KEY environment variable not set.")
if not (OPENAI_API_KEY or GOOGLE_API_KEY):
    raise ValueError("Either OPENAI_API_KEY or GOOGLE_API_KEY environment variable must be set.")

# Store loaded documents in memory for now. In a real app, use a persistent store.
# Key: document_id (e.g., hash of file content), Value: LlamaIndex Index object
document_indices = {}

# --- API Endpoints ---

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file part in the request"}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({"error": "File must be a PDF"}), 400

    try:
        # Generate a simple document ID (in a real app, use a more robust unique ID)
        # For simplicity, we'll use a hash or just the filename for now.
        # You might want to store the actual file in a 'uploads' directory
        # or a cloud storage (S3, GCS) for persistence.
        # Let's just create a temporary file in memory for LlamaParse.

        # --- PDF Processing Pipeline (using LlamaIndex and LlamaParse) ---
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.llms.openai import OpenAI # if using OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding # if using OpenAI
        from llama_index.llms.gemini import Gemini # if using Google Gemini
        from llama_index.embeddings.gemini import GeminiEmbedding # if using Google Gemini
        from llama_index.readers.llama_parse import LlamaParse
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb # Import chromadb client directly for setup
        from pathlib import Path # To manage file paths

        # Decide which LLM and Embedding model to use based on provided keys
        if OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # Set env var for LlamaIndex
            llm = OpenAI(model="gpt-3.5-turbo") # Or gpt-4
            embed_model = OpenAIEmbedding(model="text-embedding-ada-002") # Or text-embedding-3-small/large
        elif GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Set env var for LlamaIndex
            llm = Gemini(model="gemini-pro")
            embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
        else:
            return jsonify({"error": "No supported LLM/Embedding API key found"}), 500

        # Create a temporary directory for PDF processing
        temp_dir = "temp_pdf_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, pdf_file.filename)
        pdf_file.save(pdf_path) # Save the uploaded file temporarily

        # Initialize LlamaParse
        # You can set result_type="markdown" for better structure or "text"
        parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown")

        # Load the document using LlamaParse
        documents = SimpleDirectoryReader(
            input_files=[pdf_path], file_extractor={".pdf": parser}
        ).load_data()

        # Initialize ChromaDB client and collection
        db = chromadb.PersistentClient(path="./chroma_db") # Stores data in ./chroma_db folder
        chroma_collection = db.get_or_create_collection("pdf_rag_collection")

        # Create LlamaIndex vector store from ChromaDB collection
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create a VectorStoreIndex from the documents
        # The service_context will define the LLM and Embedding model
        from llama_index.core import ServiceContext
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )

        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            vector_store=vector_store,
        )

        # Store the index in our dictionary for later retrieval
        # For simplicity, let's use the filename as a document_id
        document_id = pdf_file.filename # In production, use a more robust unique ID
        document_indices[document_id] = index

        # Clean up temporary PDF file
        os.remove(pdf_path)
        if not os.listdir(temp_dir): # Remove directory if empty
            os.rmdir(temp_dir)

        return jsonify({"message": "PDF uploaded and processed successfully", "documentId": document_id}), 200

    except Exception as e:
        print(f"Error during PDF upload and processing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_with_document():
    data = request.json
    document_id = data.get('documentId')
    question = data.get('question')
    chat_history = data.get('chatHistory', []) # Optional: for conversational memory

    if not document_id or not question:
        return jsonify({"error": "documentId and question are required"}), 400

    if document_id not in document_indices:
        return jsonify({"error": "Document not found or not processed"}), 404

    index = document_indices[document_id]

    # Create a query engine
    query_engine = index.as_query_engine()

    try:
        # Query the document index
        # You might want to format chat_history into LlamaIndex message format if used for conversation
        # For simple Q&A, just the question is enough.
        response = query_engine.query(question)

        # --- Extract Citations ---
        citations = []
        for source_node in response.source_nodes:
            # LlamaIndex usually stores page_label in metadata
            page_number = source_node.metadata.get('page_label')
            # For LlamaParse, you might find more detailed source information in metadata
            # Depending on LlamaParse output, you might get a range or single page
            if page_number:
                citations.append({"pageNumber": int(page_number)}) # Ensure it's an int

        return jsonify({
            "answer": str(response),
            "citations": citations
        }), 200

    except Exception as e:
        print(f"Error during chat query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Run Flask app on port 5000
    
    
# ...
load_dotenv()

app = Flask(__name__)
CORS(app)

LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
print(f"DEBUG: LLAMAPARSE_API_KEY loaded: {LLAMAPARSE_API_KEY is not None}")
print(f"DEBUG: LLAMAPARSE_API_KEY value (partial): {LLAMAPARSE_API_KEY[:5] if LLAMAPARSE_API_KEY else 'None'}")
# ...