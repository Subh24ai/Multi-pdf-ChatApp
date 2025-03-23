# Multi-PDF Chat Assistant

![GitHub](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.25.0-red)

A powerful document intelligence system that enables users to have natural language conversations with multiple PDF documents simultaneously. This application leverages state-of-the-art NLP techniques and large language models to provide contextually-relevant answers based on the content of your documents.

![App Screenshot](/api/placeholder/800/400)

## ✨ Features

- **Multi-Document Analysis**: Upload and analyze multiple PDFs simultaneously
- **Intelligent Conversation**: Ask questions about your documents in natural language
- **Advanced Text Processing**: Extract text, tables, and images with OCR capabilities
- **Document Comparison**: Automatically identify similarities and differences between documents
- **Interactive Visualizations**: Visual representations of source documents and information retrieval
- **Customizable Models**: Configure model parameters including temperature and context size
- **Conversation Management**: Save, clear, and export your conversation history
- **Responsive UI**: Clean, modern interface with real-time progress tracking

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Database**: Facebook AI Similarity Search (FAISS)
- **Text Processing**: PyPDF2, pdfplumber, Tesseract OCR
- **LLM Integration**: Groq API with multiple model options (Llama3, Mixtral, Gemma)
- **Framework**: LangChain for retrieval-augmented generation
- **Performance**: ThreadPoolExecutor for parallel processing
- **Visualization**: Matplotlib for data representation

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-pdf-chat-assistant.git
cd multi-pdf-chat-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your API keys:
# GROQ_API_KEY=your_api_key_here
# TESSERACT_PATH=path_to_tesseract_executable
# POPPLER_PATH=path_to_poppler_bin_directory
```

## 🔧 Configuration

The application supports various configuration options:

- **Model Selection**: Choose from Llama3-70B, Llama3-8B, Mixtral-8x7B, or Gemma-7B
- **Temperature**: Adjust from 0.0 (focused) to 1.0 (creative)
- **Context Size**: Control how many text chunks are retrieved per query
- **UI Settings**: Configure maximum chat messages and other display preferences

## 💡 Usage

1. Run the application:
   ```bash
   streamlit run pp.py
   ```

2. Upload your PDF documents in the "Documents" tab
3. Click "Process Documents" to analyze and index the content
4. Switch to the "Chat" tab to start asking questions about your documents
5. Use the "Settings" tab to customize the application behavior

## 📊 How It Works

1. **Document Processing**: PDFs are uploaded and processed to extract text, tables, and images
2. **Text Chunking**: Documents are split into manageable chunks based on semantic boundaries
3. **Vectorization**: Text chunks are converted into vector embeddings using HuggingFace models
4. **Indexing**: Vectors are stored in a FAISS index for efficient similarity search
5. **Retrieval**: When a question is asked, the system finds the most relevant document chunks
6. **Generation**: The LLM uses retrieved context to generate accurate, contextual responses
7. **Citation**: Sources are tracked and displayed to enhance transparency and trust

## 🤔 Use Cases

- **Research**: Quickly extract insights from research papers and academic documents
- **Legal Document Analysis**: Navigate complex legal documents with natural language queries
- **Knowledge Management**: Create interactive knowledge bases from organizational documents
- **Educational Support**: Build learning tools that answer student questions from textbooks
- **Document Comparison**: Identify similarities and differences between multiple versions

## 🔜 Future Improvements

- [ ] Add support for more document formats (DOCX, TXT, etc.)
- [ ] Implement document-level semantic search
- [ ] Add table extraction and structured data querying
- [ ] Improve image recognition and analysis capabilities
- [ ] Enable collaborative document analysis for teams
- [ ] Add support for additional language models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the awesome web framework
- [LangChain](https://www.langchain.com/) for the RAG framework
- [HuggingFace](https://huggingface.co/) for the embedding models
- [Groq](https://groq.com/) for the LLM API access
- [FAISS](https://github.com/facebookresearch/faiss) for the vector database

---

Made with ❤️ by [Subhash Gupta](https://subh24ai.github.io/)
