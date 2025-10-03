"""
Document processing functionality for the Document Intelligence Platform
"""
import os
import tempfile
from typing import List, Optional
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import Config

class DocumentProcessor:
    """Handles document loading and processing operations"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_uploaded_documents(self, uploaded_files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
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
    def load_documents_from_directory(_self, directory_path: str = None) -> List[Document]:
        """Load and cache documents from research_papers directory"""
        if directory_path is None:
            directory_path = Config.RESEARCH_PAPERS_DIR
            
        try:
            if (os.path.exists(directory_path) and 
                any(os.path.exists(os.path.join(directory_path, f)) 
                    for f in os.listdir(directory_path))):
                loader = PyPDFDirectoryLoader(directory_path)
                docs = loader.load()
                return docs
            else:
                return []
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            final_documents = self.text_splitter.split_documents(documents)
            
            if not final_documents:
                st.error("❌ No text could be extracted from the documents!")
                return []
            
            st.info(f"📑 Created {len(final_documents)} text chunks")
            return final_documents
            
        except Exception as e:
            st.error(f"❌ Error splitting documents: {str(e)}")
            return []
    
    def validate_file_types(self, uploaded_files) -> bool:
        """Validate uploaded file types"""
        valid_extensions = tuple(f'.{ext}' for ext in Config.SUPPORTED_FILE_TYPES)
        
        for file in uploaded_files:
            if not file.name.lower().endswith(valid_extensions):
                st.error(f"❌ Unsupported file type: {file.name}")
                return False
        
        return True
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about the processed documents"""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "average_chunk_size": total_chars // len(documents) if documents else 0
        }