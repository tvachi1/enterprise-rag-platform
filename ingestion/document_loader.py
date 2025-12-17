"""
Document Loader - Enterprise-grade document ingestion with validation
"""
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document with metadata"""
    content: str
    source: str
    metadata: dict


class DocumentLoader:
    """
    Handles document loading with validation and error handling.
    Supports: PDF, TXT, Markdown
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}
    
    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
    def load(self, file_path: str) -> Optional[Document]:
        """
        Load a single document with validation
        
        Args:
            file_path: Path to document
            
        Returns:
            Document object or None if loading fails
        """
        try:
            path = Path(file_path)
            
            # Validation
            if not self._validate_file(path):
                return None
                
            # Load based on extension
            if path.suffix == '.pdf':
                content = self._load_pdf(path)
            else:
                content = self._load_text(path)
                
            if not content:
                logger.error(f"Empty content loaded from {file_path}")
                return None
                
            return Document(
                content=content,
                source=str(path),
                metadata={
                    'filename': path.name,
                    'extension': path.suffix,
                    'size_bytes': path.stat().st_size
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def load_directory(self, directory: str) -> List[Document]:
        """Load all supported documents from a directory"""
        path = Path(directory)
        documents = []
        
        for file_path in path.rglob('*'):
            if file_path.suffix in self.SUPPORTED_EXTENSIONS:
                doc = self.load(str(file_path))
                if doc:
                    documents.append(doc)
                    
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def _validate_file(self, path: Path) -> bool:
        """Validate file exists, is readable, and meets size requirements"""
        if not path.exists():
            logger.error(f"File not found: {path}")
            return False
            
        if not path.is_file():
            logger.error(f"Not a file: {path}")
            return False
            
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported extension: {path.suffix}")
            return False
            
        size = path.stat().st_size
        if size > self.max_file_size_bytes:
            logger.error(f"File too large: {size} bytes")
            return False
            
        return True
    
    def _load_pdf(self, path: Path) -> str:
        """Load PDF content using PyMuPDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
            return ""
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            return ""
    
    def _load_text(self, path: Path) -> str:
        """Load text/markdown files"""
        try:
            return path.read_text(encoding='utf-8').strip()
        except Exception as e:
            logger.error(f"Text loading error: {e}")
            return ""
