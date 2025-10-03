"""
Validation utilities for the Document Intelligence Platform
"""
import os
import re
from typing import List, Dict, Any, Optional

from config.settings import Config

class Validators:
    """Validation functions for various inputs and configurations"""
    
    @staticmethod
    def validate_api_keys() -> Dict[str, bool]:
        """Validate API keys"""
        return {
            "groq_api_key": bool(Config.GROQ_API_KEY and Config.GROQ_API_KEY.strip()),
            "openai_api_key": bool(Config.OPENAI_API_KEY and Config.OPENAI_API_KEY.strip())
        }
    
    @staticmethod
    def validate_query_input(query: str) -> Dict[str, Any]:
        """Validate user query input"""
        if not query or not query.strip():
            return {
                "valid": False,
                "error": "Query cannot be empty"
            }
        
        if len(query.strip()) < 3:
            return {
                "valid": False,
                "error": "Query must be at least 3 characters long"
            }
        
        if len(query) > 5000:
            return {
                "valid": False,
                "error": "Query is too long (max 5000 characters)"
            }
        
        return {"valid": True}
    
    @staticmethod
    def validate_file_upload(files: List) -> Dict[str, Any]:
        """Validate uploaded files"""
        if not files:
            return {
                "valid": False,
                "error": "No files uploaded"
            }
        
        errors = []
        
        for file in files:
            # Check file type
            if not file.name.lower().endswith(tuple(f'.{ext}' for ext in Config.SUPPORTED_FILE_TYPES)):
                errors.append(f"Unsupported file type: {file.name}")
                continue
            
            # Check file size
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                errors.append(f"File too large: {file.name} ({file_size_mb:.1f}MB > {Config.MAX_FILE_SIZE_MB}MB)")
        
        if errors:
            return {
                "valid": False,
                "errors": errors
            }
        
        return {"valid": True}
    
    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """Validate environment setup"""
        issues = []
        
        # Check API keys
        api_keys = Validators.validate_api_keys()
        if not api_keys["groq_api_key"]:
            issues.append("GROQ API key is missing or invalid")
        if not api_keys["openai_api_key"]:
            issues.append("OpenAI API key is missing or invalid")
        
        # Check required directories
        if not os.path.exists(Config.RESEARCH_PAPERS_DIR):
            issues.append(f"Research papers directory not found: {Config.RESEARCH_PAPERS_DIR}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "api_keys": api_keys
        }
    
    @staticmethod
    def validate_model_settings(model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Validate model configuration settings"""
        errors = []
        
        # Validate model
        from config.settings import ModelConfig
        if model not in ModelConfig.AVAILABLE_MODELS:
            errors.append(f"Invalid model: {model}")
        
        # Validate temperature
        min_temp, max_temp = ModelConfig.TEMPERATURE_RANGE
        if not (min_temp <= temperature <= max_temp):
            errors.append(f"Temperature must be between {min_temp} and {max_temp}")
        
        # Validate max tokens
        min_tokens, max_tokens_limit = ModelConfig.TOKEN_RANGE
        if not (min_tokens <= max_tokens <= max_tokens_limit):
            errors.append(f"Max tokens must be between {min_tokens} and {max_tokens_limit}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    @staticmethod
    def validate_retrieval_settings(search_type: str, k: int) -> Dict[str, Any]:
        """Validate retrieval configuration settings"""
        errors = []
        
        # Validate search type
        valid_search_types = ["similarity", "mmr", "similarity_score_threshold"]
        if search_type not in valid_search_types:
            errors.append(f"Invalid search type: {search_type}")
        
        # Validate k value
        if not (1 <= k <= 20):
            errors.append("Number of chunks to retrieve must be between 1 and 20")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    @staticmethod
    def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> Dict[str, Any]:
        """Validate document chunking settings"""
        errors = []
        
        # Validate chunk size
        if not (100 <= chunk_size <= 4000):
            errors.append("Chunk size must be between 100 and 4000 characters")
        
        # Validate chunk overlap
        if not (0 <= chunk_overlap <= chunk_size // 2):
            errors.append(f"Chunk overlap must be between 0 and {chunk_size // 2} characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters/patterns
        text = re.sub(r'[<>"\']', '', text)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def validate_session_state() -> Dict[str, Any]:
        """Validate Streamlit session state"""
        required_keys = ["chat_history"]
        optional_keys = ["vectors", "final_documents", "document_source"]
        
        import streamlit as st
        
        missing_required = [key for key in required_keys if key not in st.session_state]
        present_optional = [key for key in optional_keys if key in st.session_state]
        
        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "present_optional": present_optional,
            "has_vectors": "vectors" in st.session_state,
            "has_documents": "final_documents" in st.session_state
        }