"""
Configuration settings for the Document Intelligence Platform
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Model Settings
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 1000
    
    # Document Processing Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 10
    
    # Retrieval Settings
    DEFAULT_K = 4
    SEARCH_TYPE = "similarity"
    
    # UI Settings
    PAGE_TITLE = "Document Intelligence Platform"
    PAGE_ICON = "🔍"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # File Upload Settings
    SUPPORTED_FILE_TYPES = ['pdf', 'txt', 'doc', 'docx']
    MAX_FILE_SIZE_MB = 10
    
    # Directory Settings
    RESEARCH_PAPERS_DIR = "research_papers"
    
    # OpenMP Settings (for compatibility)
    @staticmethod
    def apply_openmp_fix():
        """Apply OpenMP fix for compatibility"""
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['OMP_NUM_THREADS'] = '1'
    
    @staticmethod
    def validate_api_keys():
        """Validate that required API keys are present"""
        missing_keys = []
        
        if not Config.GROQ_API_KEY:
            missing_keys.append("GROQ_API_KEY")
        
        if not Config.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        
        return missing_keys

class ModelConfig:
    """Model-specific configurations"""
    
    AVAILABLE_MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile", 
        "mixtral-8x7b"
    ]
    
    TEMPERATURE_RANGE = (0.0, 1.0)
    TEMPERATURE_STEP = 0.1
    
    TOKEN_RANGE = (100, 4000)
    
class EmbeddingConfig:
    """Embedding-specific configurations"""
    
    CHUNK_SIZE = 1000
    MAX_RETRIES = 3
    VECTOR_DIMENSION = 1536
    
class UIConfig:
    """UI-specific configurations"""
    
    # Theme colors
    PRIMARY_COLOR = "#00d4aa"
    SECONDARY_COLOR = "#0099cc"
    ACCENT_COLOR = "#66fcf1"
    SUCCESS_COLOR = "#90ee90"
    WARNING_COLOR = "#ffd700"
    ERROR_COLOR = "#ff4444"
    
    # Layout settings
    SIDEBAR_WIDTH = 300
    MAIN_CONTENT_WIDTH = 800