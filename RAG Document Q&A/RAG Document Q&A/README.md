# 🔍 RAG Document Q&A Platform

## Overview

This is a sophisticated **Retrieval-Augmented Generation (RAG)** document question-answering platform built with **Streamlit** and **LangChain**. The application allows users to upload documents (PDF, TXT, DOC, DOCX) or use existing research papers and ask intelligent questions about their content. It leverages OpenAI embeddings for semantic search and Groq's LLM for generating contextually relevant answers.

## 🚀 Key Features

- **📄 Multi-Format Document Support**: PDF, TXT, DOC, DOCX files
- **🧠 Intelligent Document Processing**: Automatic text chunking and preprocessing
- **🔍 Semantic Search**: OpenAI embeddings with FAISS vector store
- **💬 Interactive Chat Interface**: Streamlit-based conversational UI
- **📊 Real-time Analytics**: Query statistics and performance metrics
- **⚙️ Configurable Settings**: Customizable model parameters and retrieval settings
- **🔒 Environment Validation**: Automatic API key and dependency checks

## 🏗️ Architecture

### Project Structure
```
RAG Document Q&A/
├── app.py                     # Main application entry point
├── requirements.txt           # Python dependencies
├── research_papers/          # Default document directory
│   ├── Attention.pdf
│   ├── LLM.pdf
│   └── SQL.pdf
├── config/                   # Configuration modules
│   ├── prompts.py           # LLM prompt templates
│   └── settings.py          # Application settings
├── src/                     # Source code modules
│   ├── components/          # Core application components
│   │   ├── chat_interface.py      # Chat and query processing
│   │   ├── document_processor.py  # Document loading and processing
│   │   └── vector_store.py        # Vector database management
│   └── utils/              # Utility modules
│       ├── file_handlers.py      # File operation utilities
│       ├── ui_helpers.py         # UI component helpers
│       └── validators.py         # Input validation
└── assets/                  # UI styling
    └── styles.py           # Custom CSS styles
```

## 🔧 How I Built This Project

### 1. **Core Architecture Design**

I started by designing a modular architecture with clear separation of concerns:

- **Main Application (`app.py`)**: Entry point with the `DocumentIntelligenceApp` class
- **Configuration Layer**: Centralized settings and prompt management
- **Component Layer**: Specialized modules for different functionalities
- **Utility Layer**: Helper functions for validation, UI, and file handling

### 2. **Document Processing Pipeline**

**File: `src/components/document_processor.py`**

```python
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
```

**Key Implementation Details:**
- **Multi-format Loaders**: Used LangChain's document loaders (PyPDFLoader, TextLoader, Docx2txtLoader)
- **Intelligent Text Splitting**: RecursiveCharacterTextSplitter with configurable chunk size (1000) and overlap (200)
- **Temporary File Handling**: Safe processing of uploaded files using Python's tempfile module
- **Error Handling**: Comprehensive try-catch blocks for robust file processing

### 3. **Vector Store Management**

**File: `src/components/vector_store.py`**

```python
class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            chunk_size=EmbeddingConfig.CHUNK_SIZE,
            max_retries=EmbeddingConfig.MAX_RETRIES
        )
```

**Technical Implementation:**
- **OpenAI Embeddings**: Used text-embedding-ada-002 model for high-quality vector representations
- **FAISS Vector Store**: Facebook's efficient similarity search library for fast retrieval
- **Batch Processing**: Implemented batch processing to handle rate limits and large documents
- **Progress Tracking**: Real-time progress bars for vector creation process

### 4. **Chat Interface & RAG Chain**

**File: `src/components/chat_interface.py`**

```python
def process_query(self, user_prompt: str, vectors) -> Tuple[Optional[Dict], float]:
    # Create chains
    document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
    retriever = vectors.as_retriever(
        search_type=Config.SEARCH_TYPE,
        search_kwargs={"k": Config.DEFAULT_K}
    )
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

**RAG Implementation:**
- **Groq LLM Integration**: Used Groq's llama-3.1-8b-instant model for fast inference
- **Retrieval Chain**: LangChain's create_retrieval_chain for seamless RAG workflow
- **Context Management**: Retrieves top-k (default: 4) most relevant document chunks
- **Response Formatting**: Structured responses with source document references

### 5. **User Interface Design**

**File: `src/utils/ui_helpers.py`**

**UI Features Implemented:**
- **Multi-tab Interface**: Query, Analytics, and Settings tabs
- **Responsive Design**: Custom CSS for professional appearance
- **Real-time Feedback**: Progress bars, status indicators, and success messages
- **Sidebar Dashboard**: Document upload, system status, and quick actions
- **Analytics Dashboard**: Query statistics and performance metrics

### 6. **Configuration Management**

**File: `config/settings.py`**

```python
class Config:
    # Model Settings
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 1000
    
    # Document Processing Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Settings
    DEFAULT_K = 4
    SEARCH_TYPE = "similarity"
```

**Configuration Features:**
- **Environment Variables**: Secure API key management with python-dotenv
- **Configurable Parameters**: Easily adjustable model and processing settings
- **OpenMP Compatibility**: Fixed common threading issues with OpenMP libraries

### 7. **Validation & Error Handling**

**File: `src/utils/validators.py`**

**Validation System:**
- **Environment Validation**: Automatic API key and dependency checks
- **File Validation**: Size limits, format verification, and content validation
- **Query Validation**: Input sanitization and length checks
- **Error Reporting**: User-friendly error messages with actionable guidance

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **LLM** | Groq (Llama 3.1) | Text generation and reasoning |
| **Embeddings** | OpenAI (text-embedding-ada-002) | Semantic text representation |
| **Vector DB** | FAISS | Efficient similarity search |
| **Document Processing** | LangChain | Document loading and chunking |
| **File Handling** | PyPDF, python-docx | Multi-format document support |
| **Environment** | python-dotenv | Configuration management |

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key
- Groq API Key

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd "RAG Document Q&A"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 📖 Usage Guide

### 1. **Document Upload**
- Use the sidebar to upload PDF, TXT, DOC, or DOCX files
- Or click "USE EXISTING DOCUMENTS" to process files from `research_papers/` directory

### 2. **Ask Questions**
- Enter your question in the query interface
- Click "ANALYZE DOCUMENTS" to get AI-powered answers
- View retrieved source contexts and confidence scores

### 3. **Monitor Analytics**
- Check the Analytics tab for query statistics
- Monitor response times and document processing metrics
- Review chat history and system performance

### 4. **Customize Settings**
- Adjust model parameters (temperature, max tokens)
- Configure retrieval settings (chunk count, search type)
- Modify document processing parameters

## 🔧 Configuration Options

### Model Settings
- **LLM Model**: Choose from available Groq models
- **Temperature**: Control response creativity (0.0 - 1.0)
- **Max Tokens**: Limit response length

### Retrieval Settings
- **Chunk Count**: Number of document chunks to retrieve (1-10)
- **Search Type**: Similarity, MMR, or score threshold
- **Chunk Size**: Document splitting size (500-2000 characters)

### Document Processing
- **Chunk Overlap**: Overlap between text chunks
- **Text Splitter**: Choose splitting strategy
- **File Types**: Supported formats (PDF, TXT, DOC, DOCX)

## 🎯 Key Implementation Highlights

### 1. **Modular Architecture**
- Clean separation of concerns with dedicated modules
- Easy to maintain and extend functionality
- Scalable design for future enhancements

### 2. **Error Handling & Validation**
- Comprehensive input validation at every step
- User-friendly error messages with actionable guidance
- Graceful handling of API failures and edge cases

### 3. **Performance Optimization**
- Batch processing for large documents
- Efficient vector storage with FAISS
- Progress tracking for long-running operations

### 4. **User Experience**
- Intuitive interface with clear visual feedback
- Real-time progress indicators
- Responsive design with custom styling

### 5. **Security & Configuration**
- Secure API key management
- Environment-based configuration
- Input sanitization and validation

## 🔮 Future Enhancements

- [ ] **Multi-language Support**: Support for non-English documents
- [ ] **Advanced Retrieval**: Implement hybrid search (dense + sparse)
- [ ] **Document Management**: Persistent document storage and indexing
- [ ] **User Authentication**: Multi-user support with session management
- [ ] **Export Functionality**: Save conversations and responses
- [ ] **Advanced Analytics**: More detailed performance metrics and insights
- [ ] **API Integration**: REST API for programmatic access
- [ ] **Docker Deployment**: Containerization for easy deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer

**Syam Gudipudi**
- GitHub: [@Syam-1133](https://github.com/Syam-1133)
- Project: AI-Hand-Gesture-Controlled-Mouse

---

**Built with ❤️ using Python, Streamlit, and LangChain**