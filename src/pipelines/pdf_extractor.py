"""
PDF Text Extraction Pipeline
Extracts text content from PDF documents
"""

import os
import logging
from typing import List, Dict, Optional
from PyPDF2 import PdfReader

# Optional magic import for MIME type detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text content from PDF files"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
        self.mime_types = ['application/pdf']
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file is a supported PDF"""
        if not os.path.exists(file_path):
            return False
        
        # Check file extension
        _, ext = os.path.splitext(file_path.lower())
        if ext not in self.supported_extensions:
            return False
        
        # Check MIME type if magic is available
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(file_path, mime=True)
                return mime_type in self.mime_types
            except Exception as e:
                logger.warning(f"Could not determine MIME type for {file_path}: {e}")
                return True  # Assume it's valid if we can't check MIME type
        else:
            # Fallback: just check file extension
            return True
    
    def extract_text(self, file_path: str) -> Dict[str, any]:
        """
        Extract text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.is_supported(file_path):
            raise ValueError(f"File {file_path} is not a supported PDF")
        
        try:
            logger.info(f"Extracting text from PDF: {file_path}")
            
            # Read PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path),
                    'num_pages': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                    'author': pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                    'subject': pdf_reader.metadata.get('/Subject', '') if pdf_reader.metadata else '',
                    'creator': pdf_reader.metadata.get('/Creator', '') if pdf_reader.metadata else '',
                    'producer': pdf_reader.metadata.get('/Producer', '') if pdf_reader.metadata else '',
                    'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')) if pdf_reader.metadata else '',
                    'modification_date': str(pdf_reader.metadata.get('/ModDate', '')) if pdf_reader.metadata else ''
                }
                
                # Extract text from all pages
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        page_texts.append({
                            'page_number': page_num,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                        full_text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        page_texts.append({
                            'page_number': page_num,
                            'text': '',
                            'char_count': 0,
                            'error': str(e)
                        })
                
                result = {
                    'success': True,
                    'metadata': metadata,
                    'full_text': full_text.strip(),
                    'page_texts': page_texts,
                    'total_characters': len(full_text.strip()),
                    'extraction_method': 'PyPDF2'
                }
                
                logger.info(f"Successfully extracted {len(full_text)} characters from {metadata['num_pages']} pages")
                return result
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'file_name': os.path.basename(file_path)
            }
    
    def extract_from_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """
        Extract text from all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of extraction results for each PDF file
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        results = []
        pdf_files = [f for f in os.listdir(directory_path) 
                    if f.lower().endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            try:
                result = self.extract_text(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'file_path': file_path,
                    'file_name': pdf_file
                })
        
        return results
