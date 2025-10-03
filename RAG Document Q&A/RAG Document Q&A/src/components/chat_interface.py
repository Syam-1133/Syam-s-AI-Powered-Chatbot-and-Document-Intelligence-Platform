"""
Chat interface and query processing for the Document Intelligence Platform
"""
import time
from typing import Optional, Tuple, Dict, Any
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from datetime import datetime

from config.settings import Config
from config.prompts import PromptTemplates

class ChatInterface:
    """Handles chat interface and query processing"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt_template = PromptTemplates.get_default_qa_prompt()
    
    def _initialize_llm(self) -> Optional[ChatGroq]:
        """Initialize the Groq LLM"""
        try:
            return ChatGroq(
                groq_api_key=Config.GROQ_API_KEY, 
                model=Config.DEFAULT_MODEL, 
                temperature=Config.DEFAULT_TEMPERATURE
            )
        except Exception as e:
            st.error(f"Error initializing Groq LLM: {str(e)}")
            return None
    
    def process_query(self, user_prompt: str, vectors) -> Tuple[Optional[Dict], float]:
        """Process user query and return response"""
        if not self.llm:
            st.error("❌ LLM not initialized!")
            return None, 0
        
        try:
            with st.spinner("🔍 Searching through documents and generating response..."):
                # Create chains
                document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
                retriever = vectors.as_retriever(
                    search_type=Config.SEARCH_TYPE,
                    search_kwargs={"k": Config.DEFAULT_K}
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
    
    def display_response(self, response: Dict, response_time: float):
        """Display the query response in a formatted way"""
        if not response:
            return
        
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
            st.markdown(f'<div class="metric-card">🤖 Model Used<br><h3>{Config.DEFAULT_MODEL}</h3></div>', unsafe_allow_html=True)
        
        # Source documents
        with st.expander("📚 VIEW SOURCE DOCUMENTS", expanded=False):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Document {i+1}**")
                st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)
                if i < len(response['context']) - 1:
                    st.markdown("---")
    
    def add_to_chat_history(self, question: str, answer: str, response_time: float):
        """Add query and response to chat history"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "response_time": response_time
        })
    
    def display_chat_history(self, limit: int = 5):
        """Display recent chat history"""
        if "chat_history" not in st.session_state or not st.session_state.chat_history:
            st.info("No queries yet. Start a conversation!")
            return
        
        st.markdown("#### 💬 RECENT QUERIES")
        for i, chat in enumerate(st.session_state.chat_history[-limit:]):
            question_preview = (chat['question'][:50] + "..." 
                              if len(chat['question']) > 50 
                              else chat['question'])
            
            with st.expander(f"Q: {question_preview}", expanded=False):
                st.markdown(f'<div class="chat-bubble chat-question"><strong>Question:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                answer_preview = chat["answer"][:150] + "..." if len(chat["answer"]) > 150 else chat["answer"]
                st.markdown(f'<div class="chat-bubble chat-answer"><strong>Answer:</strong> {answer_preview}</div>', unsafe_allow_html=True)
    
    def clear_chat_history(self):
        """Clear the chat history"""
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get statistics about the chat session"""
        if "chat_history" not in st.session_state or not st.session_state.chat_history:
            return {}
        
        history = st.session_state.chat_history
        
        total_queries = len(history)
        avg_response_time = sum(chat['response_time'] for chat in history) / total_queries
        total_words_asked = sum(len(chat['question'].split()) for chat in history)
        total_words_answered = sum(len(chat['answer'].split()) for chat in history)
        
        return {
            "total_queries": total_queries,
            "avg_response_time": avg_response_time,
            "total_words_asked": total_words_asked,
            "total_words_answered": total_words_answered,
            "first_query_time": history[0]['timestamp'] if history else None,
            "last_query_time": history[-1]['timestamp'] if history else None
        }
    
    def export_chat_history(self) -> str:
        """Export chat history as formatted text"""
        if "chat_history" not in st.session_state or not st.session_state.chat_history:
            return "No chat history available."
        
        export_text = "# Document Intelligence Platform - Chat History\n\n"
        
        for i, chat in enumerate(st.session_state.chat_history, 1):
            export_text += f"## Query {i} - {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            export_text += f"**Question:** {chat['question']}\n\n"
            export_text += f"**Answer:** {chat['answer']}\n\n"
            export_text += f"**Response Time:** {chat['response_time']:.2f} seconds\n\n"
            export_text += "---\n\n"
        
        return export_text