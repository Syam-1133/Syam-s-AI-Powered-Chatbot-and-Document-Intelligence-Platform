import os
import tempfile
import shutil
# Apply OpenMP fix BEFORE importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import time
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Configure page with dark theme
st.set_page_config(
    page_title="Document Intelligence Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS with professional styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.8rem;
        color: #00d4aa;
        margin-bottom: 1rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #00d4aa, #0099cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 2px 10px rgba(0, 212, 170, 0.3);
    }
    .sub-header {
        font-size: 1.4rem;
        color: #66fcf1;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 300;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00d4aa;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #45a29e;
    }
    .success-box {
        background: linear-gradient(135deg, #1a472a, #0d1f14);
        border: 1px solid #2e8b57;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #90ee90;
    }
    .warning-box {
        background: linear-gradient(135deg, #4a3c1a, #2d240f);
        border: 1px solid #ffd700;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #ffd700;
    }
    .response-box {
        background: linear-gradient(135deg, #1a1f2e, #0f131f);
        border: 1px solid #4cc9f0;
        border-radius: 10px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        color: #e0f7fa;
        box-shadow: 0 4px 20px rgba(76, 201, 240, 0.2);
    }
    .source-box {
        background: linear-gradient(135deg, #2d3047, #1a1c2b);
        border: 1px solid #5d5f7e;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: #c5c6c7;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1f2833, #0b0c10);
        color: white;
    }
    .stButton button {
        background: linear-gradient(135deg, #00d4aa, #0099cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #0099cc, #00d4aa);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.4);
    }
    .stTextInput input, .stTextArea textarea {
        background: #1f2833 !important;
        color: white !important;
        border: 1px solid #45a29e !important;
        border-radius: 8px !important;
    }
    .stExpander {
        background: #1f2833;
        border: 1px solid #45a29e;
        border-radius: 8px;
    }
    .tab-content {
        background: #0e1117;
        padding: 1rem;
        border-radius: 10px;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #00ff00;
        box-shadow: 0 0 10px #00ff00;
    }
    .status-offline {
        background-color: #ff4444;
        box-shadow: 0 0 10px #ff4444;
    }
    .chat-bubble {
        background: linear-gradient(135deg, #2d3047, #1a1c2b);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #5d5f7e;
    }
    .chat-question {
        background: linear-gradient(135deg, #1a472a, #0d1f14);
        border-left: 4px solid #00d4aa;
    }
    .chat-answer {
        background: linear-gradient(135deg, #1a1f2e, #0f131f);
        border-left: 4px solid #4cc9f0;
    }
    .upload-section {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #45a29e;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Configure API keys
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')

# Validate API keys
if not groq_api_key:
    st.error("🔑 GROQ API key not found! Please check your .env file.")
    st.stop()

if not os.getenv('OPENAI_API_KEY'):
    st.error("🔑 OpenAI API key not found! Please check your .env file.")
    st.stop()

# Initialize LLM with error handling
@st.cache_resource
def get_llm():
    try:
        return ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0.1)
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {str(e)}")
        return None

llm = get_llm()
if llm is None:
    st.stop()

# Create enhanced prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert research assistant analyzing academic documents. 
    Provide comprehensive, accurate answers based strictly on the provided context.
    
    GUIDELINES:
    - Answer the question using only the information from the provided context
    - If the information is not in the context, clearly state this
    - Provide detailed explanations when appropriate
    - Include relevant technical details from the research papers
    - Structure your response clearly and professionally
    
    CONTEXT: {context}
    
    QUESTION: {input}
    
    Please provide a thorough, well-structured answer:
    """
)

def load_uploaded_documents(uploaded_files):
    """Load documents from uploaded files"""
    documents = []
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load based on file type
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Loaded PDF: {uploaded_file.name} ({len(docs)} pages)")
                
            elif uploaded_file.name.lower().endswith('.txt'):
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Loaded TXT: {uploaded_file.name}")
                
            elif uploaded_file.name.lower().endswith(('.doc', '.docx')):
                loader = Docx2txtLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Loaded DOCX: {uploaded_file.name}")
                
            else:
                st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    return documents

@st.cache_data
def load_documents():
    """Load and cache documents from research_papers directory"""
    try:
        if os.path.exists("research_papers") and any(os.path.exists(os.path.join("research_papers", f)) for f in os.listdir("research_papers")):
            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()
            return docs
        else:
            return []
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def create_vector_embedding_from_files(uploaded_files=None):
    """Create vector embeddings from uploaded files or existing documents"""
    try:
        # Create progress containers
        progress_container = st.container()
        status_container = st.container()
        metrics_container = st.container()
        
        with progress_container:
            st.markdown("### 📊 Document Processing Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_elapsed = st.empty()
            
        start_time = time.time()
        
        # Step 1: Load documents
        status_text.text("📄 Loading documents...")
        progress_bar.progress(10)
        
        if uploaded_files:
            # Use uploaded files
            docs = load_uploaded_documents(uploaded_files)
            source_type = "uploaded files"
        else:
            # Use existing research_papers directory
            docs = load_documents()
            source_type = "research_papers directory"
        
        if not docs:
            st.error("❌ No documents found or could be loaded!")
            return False
        
        st.info(f"📑 Loaded {len(docs)} documents from {source_type}")
        
        # Step 2: Initialize embeddings
        status_text.text("🔧 Initializing OpenAI embeddings...")
        progress_bar.progress(30)
        
        embeddings = OpenAIEmbeddings(
            chunk_size=1000,
            max_retries=3
        )
        
        # Step 3: Split documents
        status_text.text("✂️ Processing and splitting documents into chunks...")
        progress_bar.progress(50)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        final_documents = text_splitter.split_documents(docs)
        
        if not final_documents:
            st.error("❌ No text could be extracted from the documents!")
            return False
        
        st.info(f"📑 Created {len(final_documents)} text chunks")
        
        # Step 4: Create vector store
        status_text.text("🔍 Creating vector database (this may take a moment)...")
        progress_bar.progress(70)
        
        # Process in batches
        batch_size = 10
        vectors = None
        
        for i in range(0, len(final_documents), batch_size):
            batch = final_documents[i:i+batch_size]
            if vectors is None:
                vectors = FAISS.from_documents(batch, embeddings)
            else:
                batch_vectors = FAISS.from_documents(batch, embeddings)
                vectors.merge_from(batch_vectors)
            
            progress = 70 + (25 * (i + batch_size) / len(final_documents))
            progress_bar.progress(min(95, int(progress)))
            
            # Update time
            elapsed = time.time() - start_time
            time_elapsed.text(f"⏱️ Time elapsed: {elapsed:.1f} seconds")
        
        st.session_state.vectors = vectors
        st.session_state.final_documents = final_documents
        st.session_state.document_source = source_type
        
        progress_bar.progress(100)
        status_text.text("✅ Vector database created successfully!")
        
        total_time = time.time() - start_time
        
        # Display metrics
        with metrics_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Processed", len(docs))
            with col2:
                st.metric("Text Chunks Created", len(final_documents))
            with col3:
                st.metric("Processing Time", f"{total_time:.1f}s")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error creating vector embeddings: {str(e)}")
        return False

def process_query(user_prompt):
    """Process user query safely"""
    try:
        with st.spinner("🔍 Searching through documents and generating response..."):
            # Create chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Process query
            start_time = time.time()
            response = retrieval_chain.invoke({"input": user_prompt})
            end_time = time.time()
            
            return response, end_time - start_time
            
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        return None, 0

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_source" not in st.session_state:
    st.session_state.document_source = "Not initialized"

# Main UI - Professional Header
st.markdown('<div class="main-header">Syam\'s AI-Powered Document Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '🚀 AI-Powered Document Analysis & Knowledge Extraction System<br>'
    '<small style="color: #88d3ce; font-size: 1.1rem; font-weight: 300;">'
    'Upload documents • Ask questions • Get answers'
    '</small>'
    '</div>', 
    unsafe_allow_html=True
)

# Sidebar with enhanced professional layout
with st.sidebar:
    st.markdown("### 🏢 SYSTEM DASHBOARD")
    
    # System Status
    st.markdown("#### 📈 SYSTEM STATUS")
    if "vectors" in st.session_state:
        st.markdown(
            f'<div class="success-box">'
            f'<span class="status-indicator status-online"></span>'
            f'<strong>VECTOR DATABASE: ONLINE</strong><br>'
            f'Source: {st.session_state.document_source}'
            f'</div>', 
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card">📄 Document Chunks<br><h3>{len(st.session_state.final_documents)}</h3></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card">🤖 Model<br><h3>Llama-3.1-8B</h3></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="warning-box">'
            f'<span class="status-indicator status-offline"></span>'
            f'<strong>VECTOR DATABASE: OFFLINE</strong><br>'
            f'Upload documents or use existing ones'
            f'</div>', 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("#### ⚡ QUICK ACTIONS")
    
    # Document Upload Section
    st.markdown("#### 📤 UPLOAD DOCUMENTS")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'doc', 'docx'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("🚀 PROCESS UPLOADED FILES", use_container_width=True, type="primary"):
            if create_vector_embedding_from_files(uploaded_files):
                st.success("Uploaded files processed successfully!")
                st.rerun()
    
    # Use existing files button
    if st.button("USE EXISTING DOCUMENTS", use_container_width=True):
        if create_vector_embedding_from_files():
            st.success("Existing documents processed successfully!")
            st.rerun()
    
    if st.button("CLEAR CHAT HISTORY", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # Chat History
    st.markdown("#### 💬 RECENT QUERIES")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history[-5:]):
            with st.expander(f"Q: {chat['question'][:50]}..." if len(chat['question']) > 50 else f"Q: {chat['question']}", expanded=False):
                st.markdown(f'<div class="chat-bubble chat-question"><strong>Question:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bubble chat-answer"><strong>Answer:</strong> {chat["answer"][:150]}...</div>', unsafe_allow_html=True)
    else:
        st.info("No queries yet. Start a conversation!")
    
    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("• Streamlit 🎈")
    st.markdown("• LangChain ⛓️")
    st.markdown("• Groq 🚀")
    st.markdown(f"*Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 QUERY DOCUMENTS", "📊 ANALYTICS", "⚙️ SETTINGS"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 DOCUMENT QUERY INTERFACE")
        
        # System initialization status
        if "vectors" not in st.session_state:
            st.markdown(
                '<div class="upload-section">'
                '<h3>📁 Upload Documents to Get Started</h3>'
                '<p>Upload PDF, TXT, or DOCX files to analyze their content</p>'
                '<p><small>Or use the "Use Existing Documents" button if you have files in the research_papers directory</small></p>'
                '</div>', 
                unsafe_allow_html=True
            )
        else:
            # Query interface
            st.markdown("#### 📝 ASK A QUESTION")
            user_prompt = st.text_area(
                "Enter your question about the documents:",
                placeholder="e.g., What are the main findings in these documents? Explain the key concepts...",
                height=120,
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                ask_button = st.button("🔍 ANALYZE DOCUMENTS", type="primary", use_container_width=True)
            
            if ask_button and user_prompt:
                response, response_time = process_query(user_prompt)
                
                if response:
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "question": user_prompt,
                        "answer": response['answer'],
                        "timestamp": datetime.now(),
                        "response_time": response_time
                    })
                    
                    # Display response
                    st.markdown("### 📋 ANALYSIS RESULTS")
                    st.markdown(f'<div class="response-box">{response["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Response metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="metric-card">⏱️ Response Time<br><h3>{response_time:.2f}s</h3></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card">📚 Sources Retrieved<br><h3>{len(response["context"])}</h3></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-card">🤖 Model Used<br><h3>Llama-3.1-8B</h3></div>', unsafe_allow_html=True)
                    
                    # Source documents
                    with st.expander("📚 VIEW SOURCE DOCUMENTS (4 most relevant)", expanded=False):
                        for i, doc in enumerate(response['context']):
                            st.markdown(f"**Document {i+1}**")
                            st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)
                            st.markdown("---")
   

# Rest of the code remains the same for tabs 2 and 3...
with tab2:
    st.markdown("### 📊 SYSTEM ANALYTICS")
    
    if "vectors" in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 DOCUMENT STATISTICS")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f2833, #0b0c10); padding: 1.5rem; border-radius: 10px; border: 1px solid #45a29e;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                    <div style='text-align: center;'>
                        <h4 style='color: #66fcf1; margin: 0;'>Total Documents</h4>
                        <h3 style='color: #00d4aa; margin: 0.5rem 0;'>{len(st.session_state.final_documents)}</h3>
                    </div>
                    <div style='text-align: center;'>
                        <h4 style='color: #66fcf1; margin: 0;'>Text Chunks</h4>
                        <h3 style='color: #00d4aa; margin: 0.5rem 0;'>{len(st.session_state.final_documents)}</h3>
                    </div>
                    <div style='text-align: center;'>
                        <h4 style='color: #66fcf1; margin: 0;'>Document Source</h4>
                        <h3 style='color: #00d4aa; margin: 0.5rem 0; font-size: 0.9rem;'>{st.session_state.document_source}</h3>
                    </div>
                    <div style='text-align: center;'>
                        <h4 style='color: #66fcf1; margin: 0;'>Vector Dimension</h4>
                        <h3 style='color: #00d4aa; margin: 0.5rem 0;'>1536</h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ⚡ PERFORMANCE METRICS")
            if st.session_state.chat_history:
                avg_response_time = sum([chat['response_time'] for chat in st.session_state.chat_history]) / len(st.session_state.chat_history)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1f2833, #0b0c10); padding: 1.5rem; border-radius: 10px; border: 1px solid #45a29e;'>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
                        <div style='text-align: center;'>
                            <h4 style='color: #66fcf1; margin: 0;'>Total Queries</h4>
                            <h3 style='color: #00d4aa; margin: 0.5rem 0;'>{len(st.session_state.chat_history)}</h3>
                        </div>
                        <div style='text-align: center;'>
                            <h4 style='color: #66fcf1; margin: 0;'>Avg Response Time</h4>
                            <h3 style='color: #00d4aa; margin: 0.5rem 0;'>{avg_response_time:.2f}s</h3>
                        </div>
                        <div style='text-align: center; grid-column: span 2;'>
                            <h4 style='color: #66fcf1; margin: 0;'>Session Start</h4>
                            <h3 style='color: #00d4aa; margin: 0.5rem 0;'>{datetime.now().strftime('%H:%M')}</h3>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No query data available yet")
        
        # Query history
        if st.session_state.chat_history:
            st.markdown("#### 📝 QUERY HISTORY")
            history_html = """
            <div style='background: linear-gradient(135deg, #1f2833, #0b0c10); padding: 1.5rem; border-radius: 10px; border: 1px solid #45a29e;'>
                <div style='display: grid; grid-template-columns: 1fr 3fr 1fr; gap: 1rem; padding: 0.5rem; border-bottom: 1px solid #45a29e; font-weight: bold;'>
                    <div>🕒 Time</div>
                    <div>💬 Question</div>
                    <div>⚡ Response</div>
                </div>
            """
            for chat in st.session_state.chat_history[-10:]:
                question = chat['question'][:50] + "..." if len(chat['question']) > 50 else chat['question']
                history_html += f"""
                <div style='display: grid; grid-template-columns: 1fr 3fr 1fr; gap: 1rem; padding: 0.5rem; border-bottom: 1px solid #2a2d3a;'>
                    <div style='color: #90ee90;'>{chat['timestamp'].strftime('%H:%M:%S')}</div>
                    <div style='color: #e0f7fa;'>{question}</div>
                    <div style='color: #4cc9f0;'>{chat['response_time']:.2f}s</div>
                </div>
                """
            history_html += "</div>"
            st.markdown(history_html, unsafe_allow_html=True)

with tab3:
    st.markdown("### ⚙️ SYSTEM CONFIGURATION")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 MODEL SETTINGS")
        model_option = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)
        
        st.markdown("#### 🔍 RETRIEVAL SETTINGS")
        chunk_count = st.slider("Number of Chunks to Retrieve", 1, 10, 4)
        search_type = st.selectbox("Search Type", ["similarity", "mmr", "similarity_score_threshold"])
    
    with col2:
        st.markdown("#### 📄 DOCUMENT PROCESSING")
        chunk_size = st.number_input("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.number_input("Chunk Overlap", 0, 500, 200)
        text_splitter = st.selectbox("Text Splitter", ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"])
        
        st.markdown("#### ℹ️ SYSTEM INFORMATION")
        st.markdown(f'<div class="metric-card">Python Version<br><h3>3.8+</h3></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">Streamlit<br><h3>{st.__version__}</h3></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">Vector Store<br><h3>FAISS</h3></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #66fcf1; font-size: 0.9rem;'>"
    "🔍 Document Intelligence Platform • Built with Streamlit, LangChain, and Groq • "
    "Enterprise Ready AI Solution"
    "</div>", 
    unsafe_allow_html=True
)