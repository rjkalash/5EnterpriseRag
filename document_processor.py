"""
Document processing utilities for various file formats.
"""
from typing import List, Dict, Any
import os
from pathlib import Path
from loguru import logger

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


class DocumentProcessor:
    """Processes various document formats for ingestion."""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
    }
    
    def __init__(self):
        """Initialize document processor."""
        self.processed_files = []
    
    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a single file and extract text.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document dictionaries with text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        logger.info(f"Loading file: {file_path.name}")
        
        # Get appropriate loader
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        loader = loader_class(str(file_path))
        
        # Load documents
        documents = loader.load()
        
        # Format output
        processed_docs = []
        for doc in documents:
            processed_docs.append({
                "text": doc.page_content,
                "metadata": {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "extension": extension,
                    **doc.metadata,
                },
            })
        
        self.processed_files.append(str(file_path))
        logger.info(f"Loaded {len(processed_docs)} document(s) from {file_path.name}")
        
        return processed_docs
    
    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of document dictionaries
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        logger.info(f"Loading documents from: {directory_path}")
        
        all_docs = []
        
        # Get all files
        if recursive:
            files = directory_path.rglob("*")
        else:
            files = directory_path.glob("*")
        
        # Filter supported files
        supported_files = [
            f for f in files
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        
        logger.info(f"Found {len(supported_files)} supported files")
        
        # Load each file
        for file_path in supported_files:
            try:
                docs = self.load_file(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue
        
        logger.info(f"Loaded total of {len(all_docs)} documents")
        
        return all_docs
    
    def extract_text_and_metadata(
        self,
        documents: List[Dict[str, Any]],
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract texts and metadata from document dictionaries.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (texts, metadatas)
        """
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        return texts, metadatas
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_files_processed": len(self.processed_files),
            "processed_files": self.processed_files,
            "supported_extensions": list(self.SUPPORTED_EXTENSIONS.keys()),
        }
