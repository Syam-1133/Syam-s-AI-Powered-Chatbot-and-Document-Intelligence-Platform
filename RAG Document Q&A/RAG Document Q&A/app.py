"""
Main application file for the Document Intelligence Platform
Clean, modular implementation with separated concerns
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from datetime import datetime

# Import configuration
from config.settings import Config
from config.prompts import PromptTemplates

# Import components
from src.components.document_processor import DocumentProcessor
from src.components.vector_store import VectorStoreManager
from src.components.chat_interface import ChatInterface

# Import utilities
from src.utils.ui_helpers import UIHelpers
from src.utils.file_handlers import FileHandlers
from src.utils.validators import Validators

class DocumentIntelligenceApp:
    """Main application class for the Document Intelligence Platform"""
    
    def __init__(self):
        # Apply OpenMP fix
        Config.apply_openmp_fix()
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.chat_interface = ChatInterface()
        
        # Setup UI
        UIHelpers.setup_page_config()
        UIHelpers.apply_custom_css()
        
        # Initialize session state
        self._initialize_session_state()
        
        # Validate environment
        self._validate_environment()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "document_source" not in st.session_state:
            st.session_state.document_source = "Not initialized"
    
    def _validate_environment(self):
        """Validate environment and API keys"""
        validation = Validators.validate_environment()
        
        if not validation["valid"]:
            st.error("⚠️ Environment validation failed:")
            for issue in validation["issues"]:
                st.error(f"❌ {issue}")
            st.stop()
    
    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded files and create vector store"""
        # Validate files
        validation = Validators.validate_file_upload(uploaded_files)
        if not validation["valid"]:
            if "error" in validation:
                st.error(validation["error"])
            else:
                for error in validation.get("errors", []):
                    st.error(error)
            return False
        
        # Load documents
        docs = self.document_processor.load_uploaded_documents(uploaded_files)
        if not docs:
            st.error("❌ No documents could be loaded!")
            return False
        
        # Split documents
        final_documents = self.document_processor.split_documents(docs)
        if not final_documents:
            return False
        
        # Create vector store
        success, vectors = self.vector_store_manager.create_vector_store(final_documents)
        if success and vectors:
            st.session_state.vectors = vectors
            st.session_state.final_documents = final_documents
            st.session_state.document_source = "uploaded files"
            return True
        
        return False
    
    def _process_existing_documents(self):
        """Process existing documents from research_papers directory"""
        # Load documents
        docs = self.document_processor.load_documents_from_directory()
        if not docs:
            st.error("❌ No documents found in research_papers directory!")
            return False
        
        # Split documents
        final_documents = self.document_processor.split_documents(docs)
        if not final_documents:
            return False
        
        # Create vector store
        success, vectors = self.vector_store_manager.create_vector_store(final_documents)
        if success and vectors:
            st.session_state.vectors = vectors
            st.session_state.final_documents = final_documents
            st.session_state.document_source = "research_papers directory"
            return True
        
        return False
    
    def _render_sidebar(self):
        """Render the sidebar with controls and status"""
        with st.sidebar:
            st.markdown("### 🏢 SYSTEM DASHBOARD")
            
            # System Status
            UIHelpers.display_system_status()
            st.markdown("---")
            
            # Quick Actions
            st.markdown("#### ⚡ QUICK ACTIONS")
            
            # Document Upload Section
            st.markdown("#### 📤 UPLOAD DOCUMENTS")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=Config.SUPPORTED_FILE_TYPES,
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                if st.button("🚀 PROCESS UPLOADED FILES", use_container_width=True, type="primary"):
                    if self._process_uploaded_files(uploaded_files):
                        st.success("Uploaded files processed successfully!")
                        st.rerun()
            
            # Use existing files button
            if st.button("USE EXISTING DOCUMENTS", use_container_width=True):
                if self._process_existing_documents():
                    st.success("Existing documents processed successfully!")
                    st.rerun()
            
            if st.button("CLEAR CHAT HISTORY", use_container_width=True):
                self.chat_interface.clear_chat_history()
                st.rerun()
            
            st.markdown("---")
            
            # Chat History
            self.chat_interface.display_chat_history()
            
            st.markdown("---")
            UIHelpers.display_sidebar_info()
    
    def _render_query_tab(self):
        """Render the query interface tab"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 DOCUMENT QUERY INTERFACE")
            
            # System initialization status
            if "vectors" not in st.session_state:
                UIHelpers.display_upload_section()
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
                    # Validate query
                    validation = Validators.validate_query_input(user_prompt)
                    if not validation["valid"]:
                        st.error(validation["error"])
                        return
                    
                    # Process query
                    response, response_time = self.chat_interface.process_query(
                        user_prompt, 
                        st.session_state.vectors
                    )
                    
                    if response:
                        # Add to chat history
                        self.chat_interface.add_to_chat_history(
                            user_prompt, 
                            response['answer'], 
                            response_time
                        )
                        
                        # Display response
                        self.chat_interface.display_response(response, response_time)
    
    def _render_analytics_tab(self):
        """Render the analytics tab"""
        st.markdown("### 📊 SYSTEM ANALYTICS")
        
        if "vectors" in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                # Document statistics
                doc_stats = {
                    "Total Documents": len(st.session_state.final_documents),
                    "Text Chunks": len(st.session_state.final_documents),
                    "Document Source": st.session_state.document_source,
                    "Vector Dimension": "1536"
                }
                UIHelpers.display_analytics_card("📈 DOCUMENT STATISTICS", doc_stats)
            
            with col2:
                # Performance metrics
                chat_stats = self.chat_interface.get_chat_statistics()
                if chat_stats:
                    perf_stats = {
                        "Total Queries": chat_stats["total_queries"],
                        "Avg Response Time": f"{chat_stats['avg_response_time']:.2f}s",
                        "Session Start": datetime.now().strftime('%H:%M')
                    }
                    UIHelpers.display_analytics_card("⚡ PERFORMANCE METRICS", perf_stats)
                else:
                    st.info("No query data available yet")
            
            # Query history
            if st.session_state.chat_history:
                UIHelpers.display_query_history_table(st.session_state.chat_history)
        else:
            st.info("📊 Analytics will be available after processing documents")
    
    def _render_settings_tab(self):
        """Render the settings tab"""
        st.markdown("### ⚙️ SYSTEM CONFIGURATION")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 MODEL SETTINGS")
            from config.settings import ModelConfig
            
            model_option = st.selectbox("LLM Model", ModelConfig.AVAILABLE_MODELS, index=0)
            temperature = st.slider("Temperature", *ModelConfig.TEMPERATURE_RANGE, Config.DEFAULT_TEMPERATURE, ModelConfig.TEMPERATURE_STEP)
            max_tokens = st.number_input("Max Tokens", *ModelConfig.TOKEN_RANGE, Config.DEFAULT_MAX_TOKENS)
            
            st.markdown("#### 🔍 RETRIEVAL SETTINGS")
            chunk_count = st.slider("Number of Chunks to Retrieve", 1, 10, Config.DEFAULT_K)
            search_type = st.selectbox("Search Type", ["similarity", "mmr", "similarity_score_threshold"])
        
        with col2:
            st.markdown("#### 📄 DOCUMENT PROCESSING")
            chunk_size = st.number_input("Chunk Size", 500, 2000, Config.CHUNK_SIZE)
            chunk_overlap = st.number_input("Chunk Overlap", 0, 500, Config.CHUNK_OVERLAP)
            text_splitter = st.selectbox("Text Splitter", ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"])
            
            st.markdown("#### ℹ️ SYSTEM INFORMATION")
            system_info = {
                "Python Version": "3.8+",
                "Streamlit": st.__version__,
                "Vector Store": "FAISS"
            }
            for key, value in system_info.items():
                st.markdown(f'<div class="metric-card">{key}<br><h3>{value}</h3></div>', unsafe_allow_html=True)
    
    def run(self):
        """Main application entry point"""
        # Display header
        UIHelpers.display_header()
        
        # Render sidebar
        self._render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["🔍 QUERY DOCUMENTS", "📊 ANALYTICS", "⚙️ SETTINGS"])
        
        with tab1:
            self._render_query_tab()
        
        with tab2:
            self._render_analytics_tab()
        
        with tab3:
            self._render_settings_tab()
        
        # Footer
        UIHelpers.display_footer()

def main():
    """Application entry point"""
    app = DocumentIntelligenceApp()
    app.run()

if __name__ == "__main__":
    main()