# LMS AI Assistant - Text Extraction Pipeline

A comprehensive text extraction pipeline for Learning Management System (LMS) AI assistants. This system can extract text content from various sources including PDF documents, Microsoft Word documents, and YouTube videos/playlists.

## Features

- **PDF Text Extraction**: Extract text from PDF documents with metadata
- **DOC/DOCX Text Extraction**: Extract text from Microsoft Word documents
- **YouTube Transcript Extraction**: Extract transcripts from YouTube videos and playlists
- **Batch Processing**: Process multiple files and sources simultaneously
- **Unified Pipeline**: Single interface for all text extraction methods
- **Comprehensive Logging**: Detailed logging and error handling
- **Flexible Output**: Save results in structured JSON format

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LMS-AI-Assistnat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/input data/output
```

## Usage

### Command Line Interface

#### Extract from a single file:
```bash
python main.py --file document.pdf
```

#### Extract from a directory:
```bash
python main.py --directory ./documents/
```

#### Extract from YouTube video:
```bash
python main.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
```

#### Extract from YouTube playlist:
```bash
python main.py --youtube "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-videos 5
```

#### Extract from multiple sources:
```bash
python main.py --file doc1.pdf --file doc2.docx --youtube "https://youtube.com/watch?v=VIDEO_ID"
```

#### List supported formats:
```bash
python main.py --list-formats
```

### Python API

```python
from src.pipelines.text_extraction_pipeline import TextExtractionPipeline

# Initialize pipeline
pipeline = TextExtractionPipeline()

# Extract from a file
result = pipeline.extract_from_file("document.pdf")

# Extract from YouTube
result = pipeline.extract_from_youtube("https://youtube.com/watch?v=VIDEO_ID")

# Extract from directory
result = pipeline.extract_from_directory("./documents/")

# Process mixed inputs
inputs = [
    "document.pdf",
    "https://youtube.com/watch?v=VIDEO_ID"
]
result = pipeline.process_mixed_inputs(inputs)
```

## Project Structure

```
LMS-AI-Assistnat/
├── src/
│   ├── pipelines/
│   │   ├── pdf_extractor.py          # PDF text extraction
│   │   ├── doc_extractor.py          # DOC/DOCX text extraction
│   │   ├── youtube_extractor.py      # YouTube transcript extraction
│   │   └── text_extraction_pipeline.py # Unified pipeline
│   ├── config/
│   │   └── settings.py               # Configuration settings
│   └── utils/
├── data/
│   ├── input/                        # Input files directory
│   └── output/                       # Output results directory
├── main.py                           # Command line interface
├── example.py                        # Usage examples
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Supported Formats

- **PDF**: `.pdf` files
- **Microsoft Word**: `.doc`, `.docx` files
- **YouTube**: Video URLs and playlist URLs

## Configuration

The system can be configured through environment variables or by modifying `src/config/settings.py`:

- `OUTPUT_DIR`: Output directory for results (default: `data/output`)
- `INPUT_DIR`: Input directory for files (default: `data/input`)
- `LANGUAGES`: Comma-separated list of language codes for YouTube transcripts (default: `en`)

## Output Format

The pipeline generates structured JSON output containing:

- **Metadata**: File information, processing timestamps, extraction methods
- **Text Content**: Extracted text with formatting and structure
- **Statistics**: Character counts, page counts, processing statistics
- **Error Handling**: Detailed error information for failed extractions

## Examples

See `example.py` for comprehensive usage examples.

## Dependencies

- `youtube-transcript-api`: YouTube transcript extraction
- `yt-dlp`: YouTube playlist processing
- `PyPDF2`: PDF text extraction
- `python-docx`: Microsoft Word document processing
- `python-magic`: File type detection
- `requests`: HTTP requests
- `python-dotenv`: Environment variable management

## License

This project is part of the LMS AI Assistant system.

## Contributing

Please follow the existing code structure and add appropriate tests for new features.

## Support

For issues and questions, please check the logs in `data/output/extraction.log` for detailed error information.
