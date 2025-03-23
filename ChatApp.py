import streamlit as st
import os
from PyPDF2 import PdfReader
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import time
import re
import logging
from io import BytesIO
import base64
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json
from concurrent.futures import ThreadPoolExecutor

# Initialize session state variables if they do not exist
if "max_messages" not in st.session_state:
    st.session_state.max_messages = 10  # Set a default value

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Ensure chat history is also initialized

# Set the Tesseract path manually
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Configure logging with more detailed format
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Extract text, images, and tables from PDF
def extract_pdf_content(pdf_file):
    """Extract text, images, and tables from PDF"""
    text = ""
    tables = []
    images = []
    
    # Save the uploaded file to a temporary location
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
        
    # Now use the file path with pdfplumber    
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            tables.extend(page.extract_tables())
    import os
    # Extract images using pdf2image with Poppler path
    POPPLER_PATH = os.getenv("POPPLER_PATH")
    pdf_images = convert_from_path(tmp_path, poppler_path=POPPLER_PATH)
    
    for img in pdf_images:
        images.append(pytesseract.image_to_string(img))
    
    # Clean up the temporary file
    import os
    os.unlink(tmp_path)
    
    return text, tables, images

# Summarize PDF content
def summarize_text(text):
    from transformers import pipeline
    summarizer = pipeline("summarization")
    summary = summarizer(text[:1000], max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Search and filter functionality
def search_pdf(text, keyword):
    results = [line for line in text.split("\n") if keyword.lower() in line.lower()]
    return results

# Compare multiple PDFs
def compare_pdfs(pdf_texts):
    common_phrases = set.intersection(*[set(text.split()) for text in pdf_texts])
    return common_phrases

# Export chat history
def export_chat_history(chat_history):
    chat_text = json.dumps(chat_history, indent=4)
    return chat_text.encode()

# Custom CSS for better UI
def load_css():
    st.markdown("""
    <style>
        /* General Styling */
        .main {
            background-color: #f8f9fa;
            color: #212529;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Header Styling */
        .header-container {
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .app-header {
            margin: 0;
            padding: 0;
            font-weight: 700;
            font-size: 2rem;
        }
        
        /* Input Fields */
        .stTextInput > div > div > input {
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            padding: 10px 15px;
            font-size: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            border: none;
            margin: 0.5rem 0;
            width: 100%;
        }
        
        .primary-button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            color: white;
        }
        
        .primary-button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            transform: translateY(-1px);
        }
        
        .secondary-button {
            background-color: #f1f5f9;
            color: #475569;
            border: 1px solid #e2e8f0;
        }
        
        .secondary-button:hover {
            background-color: #e2e8f0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        
        .danger-button {
            background-color: #fee2e2;
            color: #ef4444;
        }
        
        .danger-button:hover {
            background-color: #fecaca;
            box-shadow: 0 2px 6px rgba(239, 68, 68, 0.15);
        }
        
        /* Chat Messages */
        .chat-container {
            padding: 1rem;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            max-height: 60vh;
            overflow-y: auto;
        }
        
        .chat-message {
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            position: relative;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chat-message-user {
            background-color: #eff6ff;
            border-left: 5px solid #3B82F6;
            margin-left: 2rem;
            margin-right: 0.5rem;
            box-shadow: 0 2px 5px rgba(59, 130, 246, 0.1);
        }
        
        .chat-message-assistant {
            background-color: #f1f5f9;
            border-left: 5px solid #22c55e;
            margin-left: 0.5rem;
            margin-right: 2rem;
            box-shadow: 0 2px 5px rgba(34, 197, 94, 0.1);
        }
        
        .chat-message-role {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: #475569;
        }
        
        .chat-message-content {
            line-height: 1.6;
            color: #334155;
        }
        
        .chat-timestamp {
            position: absolute;
            bottom: 8px;
            right: 12px;
            font-size: 0.7rem;
            color: #94a3b8;
        }
        
        /* Sources Section */
        .sources-section {
            background-color: #fefce8;
            padding: 1.2rem;
            border-radius: 12px;
            margin-top: 1rem;
            border-left: 5px solid #eab308;
            box-shadow: 0 2px 5px rgba(234, 179, 8, 0.1);
        }
        
        .sources-section h4 {
            color: #854d0e;
            margin-top: 0;
            font-size: 1rem;
            font-weight: 600;
        }
        
        /* Cards */
        .settings-card, .file-stats, .tab-content {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #f1f5f9;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            border-radius: 12px 12px 0 0;
            padding: 0 20px;
            background-color: white;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.03);
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 24px;
            border-radius: 12px 12px 0 0;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #eff6ff;
            color: #2563eb;
            border-top: 3px solid #2563eb;
            font-weight: 600;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #e2e8f0;
            border-radius: 12px;
            padding: 2rem 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #3B82F6;
            background-color: #f8fafc;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #3B82F6;
            border-radius: 10px;
        }
        
        /* Stats cards */
        .stat-card {
            background-color: white;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 4px solid #3B82F6;
            margin-bottom: 1rem;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e40af;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #64748b;
            font-weight: 500;
        }
        
        /* Form fields */
        .stSelectbox label, .stSlider label {
            font-weight: 500;
            color: #475569;
        }
        
        /* Footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 15px;
            text-align: center;
            color: white;
            font-size: 0.9rem;
            z-index: 100;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        
        .footer a {
            color: #bfdbfe;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        
        .footer a:hover {
            color: white;
            text-decoration: underline;
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem 1rem;
            background-color: #f8fafc;
            border-radius: 12px;
            border: 1px dashed #cbd5e1;
        }
        
        .empty-state-icon {
            font-size: 3rem;
            color: #94a3b8;
            margin-bottom: 1rem;
        }
        
        .empty-state-text {
            font-size: 1.1rem;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

def get_page_count(pdf):
    """Get the number of pages in a PDF file"""
    try:
        return len(PdfReader(pdf).pages)
    except Exception as e:
        logger.error(f"Error counting pages in {pdf.name}: {str(e)}")
        return 0

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files with improved error handling and text processing"""
    pdf_texts = []
    total_pages = 0
    processed_pages = 0
    
    # First pass to count total pages
    for pdf in pdf_docs:
        total_pages += get_page_count(pdf)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            metadata = {
                "name": pdf.name,
                "pages": len(pdf_reader.pages),
                "page_texts": []
            }
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or "" # Default to empty string if None
                
                # Enhanced text cleaning
                page_text = re.sub(r'\s+', ' ', page_text).strip()  # Remove extra whitespace
                page_text = re.sub(r'[^\x00-\x7F]+', ' ', page_text)  # Remove non-ASCII characters
                page_text = page_text.strip()
                
                text += page_text + " "
                metadata["page_texts"].append({
                    "page_num": i+1,
                    "text": page_text
                })
                
                processed_pages += 1
                progress_bar.progress(processed_pages / total_pages)
            
            # Basic text normalization
            text = text.strip()
            
            pdf_texts.append({
                "name": pdf.name,
                "text": text,
                "metadata": metadata
            })
            logger.info(f"Successfully processed PDF: {pdf.name} with {len(pdf_reader.pages)} pages")
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
            logger.error(f"Error processing PDF {pdf.name}: {str(e)}")
    
    progress_bar.empty()
    return pdf_texts

def get_text_chunks(pdf_texts):
    """Split text into chunks with improved parameters and metadata"""
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Adaptive chunk sizing based on PDF size
        if len(pdf_texts) > 5:
            chunk_size = 800
            chunk_overlap = 150
        else:
            chunk_size = 1000
            chunk_overlap = 200
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for i, pdf in enumerate(pdf_texts):
            pdf_chunks = text_splitter.split_text(pdf["text"])
            
            # Add detailed metadata
            for j, chunk in enumerate(pdf_chunks):
                chunks.append({
                    "chunk": chunk,
                    "source": pdf["name"],
                    "chunk_id": j,
                    "total_chunks": len(pdf_chunks),
                    "pdf_index": i
                })
                
            # Update progress bar
            progress_bar.progress((i + 1) / len(pdf_texts))
        
        progress_bar.empty()
        logger.info(f"Created {len(chunks)} text chunks from {len(pdf_texts)} PDFs")
        return chunks
    except Exception as e:
        st.error(f"Error creating text chunks: {str(e)}")
        logger.error(f"Error creating text chunks: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """Create and save vector store with improved embeddings and progress tracking"""
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        texts = [chunk["chunk"] for chunk in text_chunks]
        metadatas = [{"source": chunk["source"], 
                      "chunk_id": chunk["chunk_id"],
                      "total_chunks": chunk["total_chunks"],
                      "pdf_index": chunk["pdf_index"]} 
                     for chunk in text_chunks]
        
        # Process in batches to show progress
        batch_size = 20
        batched_texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        batched_metadatas = [metadatas[i:i + batch_size] for i in range(0, len(metadatas), batch_size)]
        
        # Initialize an empty vector store
        vector_store = None
        
        for i, (batch_texts, batch_metadatas) in enumerate(zip(batched_texts, batched_metadatas)):
            progress_text.text(f"Processing batch {i+1}/{len(batched_texts)}...")
            
            if vector_store is None:
                vector_store = FAISS.from_texts(
                    batch_texts, 
                    embedding=embeddings, 
                    metadatas=batch_metadatas
                )
            else:
                batch_vector_store = FAISS.from_texts(
                    batch_texts, 
                    embedding=embeddings, 
                    metadatas=batch_metadatas
                )
                vector_store.merge_from(batch_vector_store)
            
            # Update progress
            progress_bar.progress((i + 1) / len(batched_texts))
        
        progress_text.text("Saving vector store...")
        vector_store.save_local("faiss_index")
        
        progress_text.empty()
        progress_bar.empty()
        
        # Save additional metadata for stats
        pdf_stats = {
            "total_pdfs": len(set([chunk["pdf_index"] for chunk in text_chunks])),
            "total_chunks": len(text_chunks),
            "embedding_model": "HuggingFaceEmbeddings - MiniLM-L6-v2",
            "sources": list(set([chunk["source"] for chunk in text_chunks]))
        }
        
        # Save stats as JSON
        import json
        with open("pdf_stats.json", "w") as f:
            json.dump(pdf_stats, f)
        
        logger.info(f"Vector store created and saved with {len(texts)} embeddings")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        logger.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_prompt():
    """Get customized prompt template based on settings"""
    model_name = st.session_state.get("model_name", "llama3-70b-8192")
    
    # Base prompt template
    base_template = """
    You are an intelligent PDF Assistant that answers questions based on the provided document context.
    
    Use the following guidelines:
    1. Provide detailed and accurate information from the context.
    2. If information is not available in the context, clearly state "I don't have enough information to answer this question based on the provided documents."
    3. Cite the specific source document for your information.
    4. Present information in a clear, organized manner.
    5. If the question requires information from multiple sections, synthesize the information appropriately.
    6. Include exact page numbers when available.
    
    Context:
    {context}
    
    Chat History:
    {chat_history}
    
    Question: 
    {question}
    
    Answer:
    """
    
    # Enhanced prompt for Gemini-1.5-pro
    if "gemini-1.5-pro" in model_name:
        return """
        You are an advanced PDF research assistant with expertise in analyzing document content.
        
        Use the following guidelines:
        1. Provide comprehensive information from the document context.
        2. If information is missing from the context, clearly state "Based on the provided documents, I don't have sufficient information to answer this question completely."
        3. For each piece of information, cite the specific source document and page/section.
        4. Organize complex information into clear sections with appropriate formatting.
        5. Synthesize information from multiple sources when relevant.
        6. Be precise about what the documents state versus what might be inferred.
        7. If appropriate, suggest related questions the user might want to ask.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: 
        {question}
        
        Answer:
        """
    
    return base_template

def create_retrieval_chain(vector_store):
    """Create a retrieval chain with conversation memory and enhanced retrieval."""
    try:
        # Fetch API Key Securely
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("API key for ChatGroq is missing. Set GROQ_API_KEY in environment variables.")

        # Ensure Vector Store is Loaded
        if vector_store is None:
            st.error("Vector store is not initialized. Please process documents first.")
            return None

        # Retrieve Configurations from Streamlit Session State
        temperature = st.session_state.get("temperature", 0.2)
        k_value = st.session_state.get("k_value", 5)
        model_name = st.session_state.get("model_name", "llama3-70b-8192")

        # Set up conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Get Prompt Template
        prompt_template = get_conversation_prompt()

        # Initialize Chat Model
        model = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=2048,
        )

        # Configure Retriever with Enhanced Search
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": k_value,
                "fetch_k": min(k_value * 2, 10),  # Prevent unnecessary large fetches
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )

        # Create Conversational Retrieval Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )}
        )

        logger.info(f"✅ Retrieval chain created successfully with model={model_name}, temperature={temperature}, k={k_value}")
        return chain

    except Exception as e:
        logger.exception("Failed to create retrieval chain.")
        st.error(f"Error creating retrieval chain: {str(e)}")
        return None

def get_source_stats(source_docs):
    """Generate statistics about the sources used"""
    source_counts = {}
    for doc in source_docs:
        source = doc.metadata["source"]
        if source in source_counts:
            source_counts[source] += 1
        else:
            source_counts[source] = 1
    
    return source_counts

def render_source_citation(source_docs):
    """Render enhanced source citations with better formatting"""
    if not source_docs:
        return ""
    
    source_stats = get_source_stats(source_docs)
    
    # Sort sources by citation count
    sorted_sources = sorted(source_stats.items(), key=lambda x: x[1], reverse=True)
    
    # Create pie chart of source distribution
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = [s[0] for s in sorted_sources]
    sizes = [s[1] for s in sorted_sources]
    
    # Truncate long filenames
    labels = [l[:20] + "..." if len(l) > 20 else l for l in labels]
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    
    # Convert plot to base64 for embedding
    buf = BytesIO()
    fig.tight_layout()  # Prevents layout issues
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    # Generate source details
    source_details = ""
    for doc in source_docs:
        source = doc.metadata["source"]
        chunk_id = doc.metadata["chunk_id"]
        source_details += f"- {source} (Section {chunk_id+1})\n"
    
    return f"""
    <div class="sources-section">
        <h4>Sources Used</h4>
        <img src="data:image/png;base64,{img_str}" style="max-width: 100%;">
        <details>
            <summary>View Detailed Sources</summary>
            <pre>{source_details}</pre>
        </details>
    </div>
    """

def user_input(user_question, chat_history):
    """Process user input with improved context retrieval and answer generation"""
    try:
        start_time = time.time()
        
        if not os.path.exists("faiss_index"):
            st.error("FAISS index not found. Process documents first.")
            return chat_history

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        
        # Create retrieval chain
        chain = create_retrieval_chain(vector_store)
        
        if not chain:
            st.error("Failed to create retrieval chain")
            return chat_history
        
        # Show thinking indicator
        with st.spinner("Thinking deeply about your question..."):
            # Get response
            response = chain.invoke({"question": user_question})
            
            # Extract answer and sources
            answer = response["answer"]
            source_docs = response["source_documents"]
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_question, "timestamp": time.time()})
            chat_history.append({
                "role": "assistant", 
                "content": answer, 
                "sources": source_docs,
                "timestamp": time.time()
            })
            
            # Calculate metrics
            processing_time = time.time() - start_time
            source_count = len(set([doc.metadata["source"] for doc in source_docs]))
            
            # Log performance metrics
            logger.info(f"Response generated in {processing_time:.2f} seconds using {source_count} unique sources")
            
            return chat_history
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        logger.error(f"Error processing question: {str(e)}")
        return chat_history

def display_chat_message(message):
    """Display a single chat message with enhanced formatting"""
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message chat-message-user">
            <div class="chat-message-role">You</div>
            <div class="chat-message-content">{message['content']}</div>
            <div style="font-size: 0.8rem; color: gray; text-align: right;">
                {time.strftime('%I:%M %p', time.localtime(message.get('timestamp', time.time())))}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message chat-message-assistant">
            <div class="chat-message-role">Assistant</div>
            <div class="chat-message-content">{message['content']}</div>
            <div style="font-size: 0.8rem; color: gray; text-align: right;">
                {time.strftime('%I:%M %p', time.localtime(message.get('timestamp', time.time())))}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            st.markdown(render_source_citation(message["sources"]), unsafe_allow_html=True)

def display_chat_messages(chat_history):
    """Display chat messages with enhanced formatting"""
    for message in chat_history:
        display_chat_message(message)

def display_pdf_stats():
    """Display statistics about the processed PDFs"""
    try:
        # Load stats from JSON file
        import json
        with open("pdf_stats.json", "r") as f:
            stats = json.load(f)
        
        st.markdown("### 📊 Document Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", stats["total_pdfs"])
        with col2:
            st.metric("Total Text Chunks", stats["total_chunks"])
        
        st.markdown("#### Processed Files")
        for source in stats["sources"]:
            st.markdown(f"- {source}")
    except Exception as e:
        st.error(f"Error displaying PDF stats: {str(e)}")
        logger.error(f"Error displaying PDF stats: {str(e)}")

def show_settings_page():
    """Display and manage application settings"""
    st.markdown("## ⚙️ Application Settings")
    
    with st.form("settings_form"):
        st.markdown("### Model Settings")
        model_options = {
            "llama3-70b-8192": "Llama3 70B (Recommended)",
            "llama3-8b-8192": "Llama3 8B (Faster)",
            "mixtral-8x7b-32768": "Mixtral 8x7B (Balanced)",
            "gemma-7b-it": "Gemma 7B (Smaller)"
        }
        
        model_name = st.selectbox(
            "Select LLM Model", 
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.get("temperature", 0.2), 
            step=0.1,
            help="Lower values for more focused, accurate responses; higher values for more creative responses"
        )
        
        k_value = st.slider(
            "Number of Source Chunks", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.get("k_value", 5), 
            step=1,
            help="How many text chunks to retrieve for context. Higher values may improve comprehensiveness but could introduce noise"
        )
        
        st.markdown("### UI Settings")
        max_messages = st.number_input(
            "Maximum Chat Messages", 
            min_value=10, 
            max_value=100, 
            value=st.session_state.get("max_messages", 50),
            help="Maximum number of messages to display in the chat history"
        )
        
        submit = st.form_submit_button("Save Settings")
        
        if submit:
            st.session_state.temperature = temperature
            st.session_state.k_value = k_value
            st.session_state.model_name = model_name
            st.session_state.max_messages = max_messages
            st.success("Settings saved successfully!")

def export_chat():
    """Export chat history to a file"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export")
        return
    
    chat_text = "# PDF Chat Assistant - Conversation Log\n\n"
    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        timestamp = time.strftime('%Y-%m-%d %I:%M %p', time.localtime(msg.get('timestamp', time.time())))
        
        chat_text += f"## {role} ({timestamp})\n\n"
        chat_text += f"{content}\n\n"
        
        # Add source information for assistant messages
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            chat_text += "### Sources:\n\n"
            unique_sources = set()
            for doc in msg["sources"]:
                source = doc.metadata["source"]
                chunk_id = doc.metadata["chunk_id"]
                if source not in unique_sources:
                    chat_text += f"- {source} (Section {chunk_id+1})\n"
                    unique_sources.add(source)
            chat_text += "\n"
    
    # Convert to bytes for download
    chat_bytes = chat_text.encode()
    
    return chat_bytes

def main():
    # Page configuration
    st.set_page_config(
        page_title="Multi-PDFs Chat Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False

    # App header
    st.markdown("""
    <div class="header-container">
        <h1 class="app-header">📚 Multi-PDFs Chat Assistant</h1>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "⚙️ Settings"])

    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        if not st.session_state.vector_store_created:
            st.info("Please upload and process documents in the Documents tab before starting a chat.")
        else:
            chat_container = st.container()
            with chat_container:
                display_chat_messages(st.session_state.chat_history[-st.session_state.max_messages:])

            st.markdown("---")
            user_question = st.text_input("Ask a question about your documents:", key="user_question")

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button("Send", key="send_button"):
                    if user_question:
                        st.session_state.chat_history = user_input(user_question, st.session_state.chat_history)
                        if "user_question" not in st.session_state:
                            st.session_state["user_question"] = ""
                        st.rerun()

            with col2:
                if st.button("Clear Chat", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()

            with col3:
                if st.button("Export Chat", key="export_chat"):
                    chat_bytes = export_chat()
                    if chat_bytes:
                        st.download_button(
                            label="Download Conversation",
                            data=chat_bytes,
                            file_name=f"pdf_chat_{time.strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("## 📁 Document Management")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            help="Upload one or more PDF documents to analyze"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Process Documents", key="process_docs"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                else:
                    st.session_state.vector_store_created = False
                    with st.spinner("Processing documents..."):
                        st.session_state.chat_history = []
                        
                        # Parallel PDF processing
                        def process_pdf(pdf_file):
                            return extract_pdf_content(pdf_file)

                        with ThreadPoolExecutor() as executor:
                            pdf_results = list(executor.map(process_pdf, pdf_docs))

                        # Extract text, tables, and images
                        pdf_texts = [result[0] for result in pdf_results]  # Extract only text
                        pdf_tables = [result[1] for result in pdf_results]  # Extract tables
                        pdf_images = [result[2] for result in pdf_results]  # Extract images
                        
                        # Display summary for each document
                        for pdf_file, text, tables, images in zip(pdf_docs, pdf_texts, pdf_tables, pdf_images):
                            summary = summarize_text(text)  # Fixed missing 'text' issue
                            st.write(f"**Summary of {pdf_file.name}:**")
                            st.write(summary)

                            keyword = st.text_input(f"Search within {pdf_file.name}")
                            if keyword:
                                search_results = search_pdf(text, keyword)
                                st.write("Search Results:", search_results)
                        
                        # Compare PDFs if multiple are uploaded
                        if len(pdf_texts) > 1:
                            common_content = compare_pdfs(pdf_texts)
                            st.write("Common Topics Across PDFs:", common_content)
                            
                        # Chunk the extracted text    
                        text_chunks = get_text_chunks([{"name": f.name, "text": t} for f, t in zip(pdf_docs, pdf_texts)])
                        
                        # Create vector store
                        if text_chunks:
                            vector_store = get_vector_store(text_chunks)
                            if vector_store:
                                st.session_state.vector_store_created = True
                                st.success(f"✅ Successfully processed {len(pdf_docs)} documents with {len(text_chunks)} chunks")
                                st.balloons()

        with col2:
            if st.button("Reset Knowledge Base", key="reset_kb"):
                if os.path.exists("faiss_index"):
                    import shutil
                    shutil.rmtree("faiss_index")
                    if os.path.exists("pdf_stats.json"):
                        os.remove("pdf_stats.json")
                    st.session_state.vector_store_created = False
                    st.session_state.chat_history = []
                    st.success("Knowledge base has been reset. You can upload new documents.")
                    st.rerun()

        if st.session_state.vector_store_created and os.path.exists("pdf_stats.json"):
            display_pdf_stats()

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        show_settings_page()
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #1E3A5F; padding: 15px; text-align: center; color: white;">
            © <a href="https://subh24ai.github.io" target="_blank" style="color: #4CAF50;">Subhash Gupta</a> | Made with ❤️ | Chat with Multi_PDFs
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()