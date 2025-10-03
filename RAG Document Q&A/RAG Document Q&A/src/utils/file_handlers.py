"""
File handling utilities for the Document Intelligence Platform
"""
import os
import shutil
from typing import List, Dict, Any
from pathlib import Path

from config.settings import Config

class FileHandlers:
    """Utility functions for file operations"""
    
    @staticmethod
    def validate_file_size(file, max_size_mb: int = None) -> bool:
        """Validate file size"""
        max_size_mb = max_size_mb or Config.MAX_FILE_SIZE_MB
        max_size_bytes = max_size_mb * 1024 * 1024
        
        file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else file.size
        return file_size <= max_size_bytes
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension"""
        return Path(filename).suffix.lower()
    
    @staticmethod
    def is_supported_file_type(filename: str) -> bool:
        """Check if file type is supported"""
        extension = FileHandlers.get_file_extension(filename).lstrip('.')
        return extension in Config.SUPPORTED_FILE_TYPES
    
    @staticmethod
    def create_backup_directory(base_path: str) -> str:
        """Create backup directory"""
        backup_dir = os.path.join(base_path, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        return backup_dir
    
    @staticmethod
    def backup_file(file_path: str, backup_dir: str) -> str:
        """Create backup of a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        filename = os.path.basename(file_path)
        backup_path = os.path.join(backup_dir, f"{filename}.backup")
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    @staticmethod
    def get_directory_info(directory_path: str) -> Dict[str, Any]:
        """Get information about a directory"""
        if not os.path.exists(directory_path):
            return {"exists": False}
        
        files = []
        total_size = 0
        
        for root, dirs, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    files.append({
                        "name": filename,
                        "path": file_path,
                        "size": file_size,
                        "extension": FileHandlers.get_file_extension(filename),
                        "is_supported": FileHandlers.is_supported_file_type(filename)
                    })
                except OSError:
                    continue
        
        return {
            "exists": True,
            "total_files": len(files),
            "total_size": total_size,
            "files": files,
            "supported_files": [f for f in files if f["is_supported"]]
        }
    
    @staticmethod
    def clean_temp_files(temp_dir: str = None):
        """Clean temporary files"""
        if temp_dir is None:
            temp_dir = "/tmp"
        
        try:
            temp_files = [f for f in os.listdir(temp_dir) if f.startswith("tmp")]
            for temp_file in temp_files:
                temp_path = os.path.join(temp_dir, temp_file)
                try:
                    if os.path.isfile(temp_path):
                        os.unlink(temp_path)
                    elif os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                except OSError:
                    continue
        except OSError:
            pass
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create a safe filename by removing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        safe_name = filename
        
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '_')
        
        return safe_name
    
    @staticmethod
    def get_file_stats(files: List) -> Dict[str, Any]:
        """Get statistics about uploaded files"""
        if not files:
            return {}
        
        total_size = sum(len(f.getvalue()) for f in files)
        file_types = {}
        
        for file in files:
            ext = FileHandlers.get_file_extension(file.name)
            file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": file_types,
            "average_size_mb": (total_size / len(files)) / (1024 * 1024)
        }