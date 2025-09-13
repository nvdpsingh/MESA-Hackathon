"""
Unified Text Extraction Pipeline
Main pipeline class that coordinates all text extraction methods
"""

import os
import logging
from typing import List, Dict, Union, Optional, Any
from datetime import datetime

from .pdf_extractor import PDFExtractor
from .doc_extractor import DOCExtractor
from .youtube_extractor import YouTubeExtractor

logger = logging.getLogger(__name__)

class TextExtractionPipeline:
    """
    Unified pipeline for extracting text from various sources:
    - PDF documents
    - DOC/DOCX documents  
    - YouTube videos and playlists
    """
    
    def __init__(self, output_dir: str = "data/output"):
        """
        Initialize the text extraction pipeline
        
        Args:
            output_dir: Directory to save extracted text files
        """
        self.output_dir = output_dir
        self.pdf_extractor = PDFExtractor()
        self.doc_extractor = DOCExtractor()
        self.youtube_extractor = YouTubeExtractor()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'extraction.log')),
                logging.StreamHandler()
            ]
        )
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a single file (PDF, DOC, DOCX)
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing extraction results
        """
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f'File not found: {file_path}',
                'file_path': file_path
            }
        
        file_ext = os.path.splitext(file_path.lower())[1]
        
        try:
            if file_ext == '.pdf':
                result = self.pdf_extractor.extract_text(file_path)
            elif file_ext in ['.doc', '.docx']:
                result = self.doc_extractor.extract_text(file_path)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'file_path': file_path
                }
            
            # Add pipeline metadata
            result['pipeline_metadata'] = {
                'extracted_at': datetime.now().isoformat(),
                'pipeline_version': '1.0.0',
                'extractor_type': 'file'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def extract_from_youtube(self, url: str, languages: List[str] = ['en'], 
                           max_videos: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract text from YouTube video or playlist
        
        Args:
            url: YouTube video or playlist URL
            languages: List of language codes to try
            max_videos: Maximum number of videos to process (for playlists)
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            result = self.youtube_extractor.extract_from_url(url, languages)
            
            # Add pipeline metadata
            result['pipeline_metadata'] = {
                'extracted_at': datetime.now().isoformat(),
                'pipeline_version': '1.0.0',
                'extractor_type': 'youtube',
                'languages_requested': languages,
                'max_videos_limit': max_videos
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing YouTube URL {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def extract_from_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Extract text from all supported files in a directory
        
        Args:
            directory_path: Path to directory containing files
            
        Returns:
            Dictionary containing extraction results for all files
        """
        if not os.path.exists(directory_path):
            return {
                'success': False,
                'error': f'Directory not found: {directory_path}',
                'directory_path': directory_path
            }
        
        # Find all supported files
        supported_files = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file.lower())[1]
                if file_ext in ['.pdf', '.doc', '.docx']:
                    supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
        
        # Process each file
        results = []
        successful_extractions = 0
        failed_extractions = 0
        
        for file_path in supported_files:
            result = self.extract_from_file(file_path)
            results.append(result)
            
            if result['success']:
                successful_extractions += 1
            else:
                failed_extractions += 1
        
        # Create summary
        summary = {
            'directory_path': directory_path,
            'total_files': len(supported_files),
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': (successful_extractions / len(supported_files)) * 100 if supported_files else 0
        }
        
        return {
            'success': True,
            'summary': summary,
            'files': results,
            'pipeline_metadata': {
                'extracted_at': datetime.now().isoformat(),
                'pipeline_version': '1.0.0',
                'extractor_type': 'directory'
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save extraction results to a file
        
        Args:
            results: Extraction results dictionary
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extraction_results_{timestamp}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get list of supported file formats
        
        Returns:
            Dictionary mapping extractor types to supported formats
        """
        return {
            'pdf': self.pdf_extractor.supported_extensions,
            'doc': self.doc_extractor.supported_extensions,
            'youtube': ['video_url', 'playlist_url']
        }
    
    def process_mixed_inputs(self, inputs: List[Union[str, Dict[str, str]]]) -> Dict[str, Any]:
        """
        Process a mixed list of files and YouTube URLs
        
        Args:
            inputs: List of file paths or dictionaries with 'type' and 'url' keys
            
        Returns:
            Dictionary containing all extraction results
        """
        results = {
            'files': [],
            'youtube': [],
            'summary': {
                'total_inputs': len(inputs),
                'successful_extractions': 0,
                'failed_extractions': 0
            },
            'pipeline_metadata': {
                'extracted_at': datetime.now().isoformat(),
                'pipeline_version': '1.0.0',
                'extractor_type': 'mixed'
            }
        }
        
        for i, input_item in enumerate(inputs):
            if isinstance(input_item, str):
                # Assume it's a file path
                if os.path.exists(input_item) and os.path.isfile(input_item):
                    result = self.extract_from_file(input_item)
                    results['files'].append(result)
                else:
                    # Assume it's a YouTube URL
                    result = self.extract_from_youtube(input_item)
                    results['youtube'].append(result)
            elif isinstance(input_item, dict):
                # Dictionary with type and url/path
                input_type = input_item.get('type', 'file')
                input_path = input_item.get('url') or input_item.get('path', '')
                
                if input_type == 'youtube':
                    result = self.extract_from_youtube(input_path)
                    results['youtube'].append(result)
                else:
                    result = self.extract_from_file(input_path)
                    results['files'].append(result)
            
            # Update summary
            if result.get('success', False):
                results['summary']['successful_extractions'] += 1
            else:
                results['summary']['failed_extractions'] += 1
        
        results['summary']['success_rate'] = (
            results['summary']['successful_extractions'] / results['summary']['total_inputs'] * 100
            if results['summary']['total_inputs'] > 0 else 0
        )
        
        return results
