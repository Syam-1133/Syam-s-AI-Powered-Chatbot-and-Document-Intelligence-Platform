"""
Vector store management for the Document Intelligence Platform
"""
import time
from typing import List, Optional, Tuple
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config.settings import Config, EmbeddingConfig

class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            chunk_size=EmbeddingConfig.CHUNK_SIZE,
            max_retries=EmbeddingConfig.MAX_RETRIES
        )
        self.batch_size = Config.BATCH_SIZE
    
    def create_vector_store(self, documents: List[Document]) -> Tuple[bool, Optional[FAISS]]:
        """Create vector store from documents with progress tracking"""
        try:
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            metrics_container = st.container()
            
            with progress_container:
                st.markdown("### 📊 Vector Store Creation Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_elapsed = st.empty()
                
            start_time = time.time()
            
            # Step 1: Initialize embeddings
            status_text.text("🔧 Initializing OpenAI embeddings...")
            progress_bar.progress(10)
            
            # Step 2: Create vector store
            status_text.text("🔍 Creating vector database (this may take a moment)...")
            progress_bar.progress(30)
            
            # Process in batches to avoid rate limits
            vectors = None
            total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i+self.batch_size]
                batch_num = i // self.batch_size + 1
                
                status_text.text(f"🔍 Processing batch {batch_num}/{total_batches}...")
                
                if vectors is None:
                    vectors = FAISS.from_documents(batch, self.embeddings)
                else:
                    batch_vectors = FAISS.from_documents(batch, self.embeddings)
                    vectors.merge_from(batch_vectors)
                
                # Update progress
                progress = 30 + (60 * batch_num / total_batches)
                progress_bar.progress(min(90, int(progress)))
                
                # Update time
                elapsed = time.time() - start_time
                time_elapsed.text(f"⏱️ Time elapsed: {elapsed:.1f} seconds")
            
            # Final steps
            progress_bar.progress(100)
            status_text.text("✅ Vector database created successfully!")
            
            total_time = time.time() - start_time
            
            # Display metrics
            with metrics_container:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents Processed", len(documents))
                with col2:
                    st.metric("Vector Dimension", EmbeddingConfig.VECTOR_DIMENSION)
                with col3:
                    st.metric("Processing Time", f"{total_time:.1f}s")
            
            return True, vectors
            
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            return False, None
    
    def save_vector_store(self, vectors: FAISS, path: str) -> bool:
        """Save vector store to disk"""
        try:
            vectors.save_local(path)
            st.success(f"✅ Vector store saved to {path}")
            return True
        except Exception as e:
            st.error(f"❌ Error saving vector store: {str(e)}")
            return False
    
    def load_vector_store(self, path: str) -> Optional[FAISS]:
        """Load vector store from disk"""
        try:
            vectors = FAISS.load_local(path, self.embeddings)
            st.success(f"✅ Vector store loaded from {path}")
            return vectors
        except Exception as e:
            st.error(f"❌ Error loading vector store: {str(e)}")
            return None
    
    def get_retriever(self, vectors: FAISS, search_type: str = None, k: int = None):
        """Get retriever from vector store"""
        search_type = search_type or Config.SEARCH_TYPE
        k = k or Config.DEFAULT_K
        
        return vectors.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def similarity_search(self, vectors: FAISS, query: str, k: int = None) -> List[Document]:
        """Perform similarity search"""
        k = k or Config.DEFAULT_K
        return vectors.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, vectors: FAISS, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores"""
        k = k or Config.DEFAULT_K
        return vectors.similarity_search_with_score(query, k=k)
    
    def get_vector_store_info(self, vectors: FAISS) -> dict:
        """Get information about the vector store"""
        try:
            # Get the number of vectors
            vector_count = vectors.index.ntotal if hasattr(vectors.index, 'ntotal') else "Unknown"
            
            return {
                "vector_count": vector_count,
                "embedding_dimension": EmbeddingConfig.VECTOR_DIMENSION,
                "index_type": type(vectors.index).__name__
            }
        except Exception as e:
            st.error(f"Error getting vector store info: {str(e)}")
            return {}