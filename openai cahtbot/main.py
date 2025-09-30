import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are Syam AI Assistant, a professional AI assistant designed for workplace productivity. 
        Provide clear, detailed, and helpful responses. Be conversational but maintain professionalism. 
        Always aim to be accurate, helpful, and efficient in your responses."""),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    try:
        openai.api_key = api_key
        llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="Syam's AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for exact screenshot styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    
    /* EXACT SCREENSHOT STYLING */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 20px;
        background-color: white;
    }
    .user-message-container {
        margin: 15px 0;
        text-align: right;
    }
    .assistant-message-container {
        margin: 15px 0;
        text-align: left;
    }
    .user-label {
        font-weight: bold;
        color: #000000;
        margin-bottom: 5px;
        font-size: 14px;
    }
    .assistant-label {
        font-weight: bold;
        color: #000000;
        margin-bottom: 5px;
        font-size: 14px;
    }
    .user-message {
        background-color: #f0f0f0;
        color: #000000 !important;
        padding: 12px 16px;
        border-radius: 10px;
        display: inline-block;
        max-width: 70%;
        text-align: left;
        border: 1px solid #ddd;
    }
    .assistant-message {
        background-color: #f0f0f0;
        color: #000000 !important;
        padding: 12px 16px;
        border-radius: 10px;
        display: inline-block;
        max-width: 70%;
        text-align: left;
        border: 1px solid #ddd;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .api-key-info {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Syam\'s AI Chatbot</h1>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # API Key section with better instructions
    st.markdown("### 🔑 API Key Configuration")
    
 
    st.markdown("**How to get your API key:**")
    st.markdown("1. Visit [OpenAI Platform](https://platform.openai.com/)")
    st.markdown("2. Sign up or log in to your account")
    st.markdown("3. Go to API Keys section")
    st.markdown("4. Create a new API key")
    st.markdown("</div>", unsafe_allow_html=True)
    
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", 
                           placeholder="sk-...",
                           help="Your API key is never stored and is only used for the current session")
    
    # Show API key status
    if api_key:
        st.success("✅ API Key provided")
    else:
        st.warning("⚠️ API Key required to use the chatbot")
    
    # Model selection
    st.markdown("### 🧠 Model Configuration")
    engine = st.selectbox("Select AI Model", 
                         ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                         help="Choose the AI model for generating responses")
    
    # Response parameters
    st.markdown("### ⚡ Response Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature", 
                               min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                               help="Lower values make responses more deterministic")
    
    with col2:
        max_tokens = st.slider("Max Tokens", 
                              min_value=50, max_value=500, value=150, step=50,
                              help="Maximum length of the response")
    
    # Info section
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI assistant uses OpenAI's powerful language models to answer your questions. 
    Adjust the settings to customize the response behavior.
    """)

# Main chat interface
st.markdown("## 💬 Chat with Syam's AI chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display API key requirement message if no key is provided
if not api_key:
    st.markdown("---")
    st.markdown("""
    ## 🔑 API Key Required
    
    To start chatting with the AI assistant, you need to:
    
    1. **Get an OpenAI API key** from [OpenAI Platform](https://platform.openai.com/)
    2. **Enter your API key** in the sidebar on the left
    3. **Start chatting** by typing your question below
    
    🔒 **Your API key is secure** - it's only used for this session and never stored.
    """)
    st.markdown("---")
    
    # Demo section to show how it works
    st.markdown("---")
    st.markdown("### 🎯 How it works:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Get API Key**")
        st.markdown("Sign up at OpenAI and get your free API key with credits")
    
    with col2:
        st.markdown("**2. Enter Key**")
        st.markdown("Paste your API key in the sidebar field")
    
    with col3:
        st.markdown("**3. Start Chatting**")
        st.markdown("Ask any question and get AI-powered responses")

# Display chat messages only if API key is provided
if api_key:
    # Display chat messages
    if st.session_state.messages:
        st.markdown("---")
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    '''
                    <div class="user-message-container">
                        <div class="user-label">YOU</div>
                        <div class="user-message">{}</div>
                    </div>
                    '''.format(message["content"]), 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '''
                    <div class="assistant-message-container">
                        <div class="assistant-label">AI ASSISTANT</div>
                        <div class="assistant-message">{}</div>
                    </div>
                    '''.format(message["content"]), 
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown("""
        ## 🚀 Ready to Chat!

        Your API key is configured and you're all set to start chatting with the Syam's AI Chatbot.

        **Type your question below** and press Enter to get started!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Chat input (only show if API key is provided)
if api_key:
    user_input = st.chat_input("Ask me anything...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate and display assistant response
        with st.spinner(" Syam's AI is Thinking...🤔"):
            response = generate_response(user_input, api_key, engine, temperature, max_tokens)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display updated chat
        st.rerun()

# Clear chat button (only show if there are messages)
if api_key and st.session_state.messages:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by OpenAI and Streamlit | Built with ❤️"
    "</div>", 
    unsafe_allow_html=True
)