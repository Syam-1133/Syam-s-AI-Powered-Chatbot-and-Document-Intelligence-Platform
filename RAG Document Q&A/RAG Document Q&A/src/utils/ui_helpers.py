"""
UI helper functions for the Document Intelligence Platform
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Any

from config.settings import Config
from assets.styles import (
    get_dark_theme_css, 
    get_analytics_card_css, 
    get_performance_metrics_css,
    get_query_history_css,
    get_footer_css
)

class UIHelpers:
    """Helper functions for UI components"""
    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT,
            initial_sidebar_state=Config.INITIAL_SIDEBAR_STATE
        )
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling"""
        st.markdown(get_dark_theme_css(), unsafe_allow_html=True)
    
    @staticmethod
    def display_header():
        """Display the main header"""
        st.markdown(
            '<div class="main-header">Syam\'s AI-Powered Document Analyzer</div>', 
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">'
            '🚀 AI-Powered Document Analysis & Knowledge Extraction System<br>'
            '<small style="color: #88d3ce; font-size: 1.1rem; font-weight: 300;">'
            'Upload documents • Ask questions • Get answers'
            '</small>'
            '</div>', 
            unsafe_allow_html=True
        )
    
    @staticmethod
    def display_system_status():
        """Display system status in sidebar"""
        st.markdown("#### 📈 SYSTEM STATUS")
        
        if "vectors" in st.session_state:
            st.markdown(
                f'<div class="success-box">'
                f'<span class="status-indicator status-online"></span>'
                f'<strong>VECTOR DATABASE: ONLINE</strong><br>'
                f'Source: {st.session_state.get("document_source", "Unknown")}'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                doc_count = len(st.session_state.get("final_documents", []))
                st.markdown(
                    f'<div class="metric-card">📄 Document Chunks<br><h3>{doc_count}</h3></div>', 
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card">🤖 Model<br><h3>{Config.DEFAULT_MODEL}</h3></div>', 
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f'<div class="warning-box">'
                f'<span class="status-indicator status-offline"></span>'
                f'<strong>VECTOR DATABASE: OFFLINE</strong><br>'
                f'Upload documents or use existing ones'
                f'</div>', 
                unsafe_allow_html=True
            )
    
    @staticmethod
    def display_upload_section():
        """Display document upload section"""
        if "vectors" not in st.session_state:
            st.markdown(
                '<div class="upload-section">'
                '<h3>📁 Upload Documents to Get Started</h3>'
                '<p>Upload PDF, TXT, or DOCX files to analyze their content</p>'
                '<p><small>Or use the "Use Existing Documents" button if you have files in the research_papers directory</small></p>'
                '</div>', 
                unsafe_allow_html=True
            )
    
    @staticmethod
    def display_metrics(metrics: Dict[str, Any]):
        """Display metrics in a formatted grid"""
        if not metrics:
            return
        
        if len(metrics) == 3:
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
        elif len(metrics) == 2:
            col1, col2 = st.columns(2)
            cols = [col1, col2]
        else:
            cols = [st.container()]
        
        for i, (key, value) in enumerate(metrics.items()):
            if i < len(cols):
                with cols[i]:
                    st.metric(key.replace('_', ' ').title(), value)
    
    @staticmethod
    def display_analytics_card(title: str, data: Dict[str, Any]):
        """Display analytics data in a card format"""
        st.markdown(f"#### {title}")
        
        card_html = get_analytics_card_css()
        
        for key, value in data.items():
            card_html += f"""
            <div style='text-align: center;'>
                <h4 style='color: #66fcf1; margin: 0;'>{key.replace('_', ' ').title()}</h4>
                <h3 style='color: #00d4aa; margin: 0.5rem 0;'>{value}</h3>
            </div>
            """
        
        card_html += """
                </div>
            </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def display_query_history_table(chat_history, limit: int = 10):
        """Display chat history in a table format"""
        if not chat_history:
            st.info("No query data available yet")
            return
        
        st.markdown("#### 📝 QUERY HISTORY")
        
        history_html = get_query_history_css()
        
        for chat in chat_history[-limit:]:
            question = (chat['question'][:50] + "..." 
                       if len(chat['question']) > 50 
                       else chat['question'])
            
            history_html += f"""
            <div style='display: grid; grid-template-columns: 1fr 3fr 1fr; gap: 1rem; padding: 0.5rem; border-bottom: 1px solid #2a2d3a;'>
                <div style='color: #90ee90;'>{chat['timestamp'].strftime('%H:%M:%S')}</div>
                <div style='color: #e0f7fa;'>{question}</div>
                <div style='color: #4cc9f0;'>{chat['response_time']:.2f}s</div>
            </div>
            """
        
        history_html += "</div>"
        st.markdown(history_html, unsafe_allow_html=True)
    
    @staticmethod
    def display_footer():
        """Display the footer"""
        st.markdown("---")
        st.markdown(get_footer_css(), unsafe_allow_html=True)
    
    @staticmethod
    def display_sidebar_info():
        """Display information in sidebar"""
        st.markdown("---")
        st.markdown("**Built with:**")
        st.markdown("• Streamlit 🎈")
        st.markdown("• LangChain ⛓️")
        st.markdown("• Groq 🚀")
        st.markdown(f"*Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    @staticmethod
    def create_download_button(data: str, filename: str, label: str):
        """Create a download button for data"""
        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime="text/plain"
        )