#!/usr/bin/env python3
"""
Example usage of the LMS AI Assistant Text Extraction Pipeline
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipelines.text_extraction_pipeline import TextExtractionPipeline

def example_single_file():
    """Example: Extract text from a single PDF file"""
    print("=== Single File Extraction Example ===")
    
    pipeline = TextExtractionPipeline()
    
    # Example with a PDF file (replace with actual file path)
    file_path = "data/input/sample.pdf"
    
    if os.path.exists(file_path):
        result = pipeline.extract_from_file(file_path)
        
        if result['success']:
            print(f"✓ Successfully extracted text from {file_path}")
            print(f"  Characters: {result['total_characters']}")
            print(f"  Pages: {result['metadata']['num_pages']}")
        else:
            print(f"✗ Failed to extract text: {result['error']}")
    else:
        print(f"File not found: {file_path}")
        print("Please add a sample PDF file to data/input/")

def example_youtube_video():
    """Example: Extract transcript from a YouTube video"""
    print("\n=== YouTube Video Extraction Example ===")
    
    pipeline = TextExtractionPipeline()
    
    # Example YouTube video URL
    video_url = "https://www.youtube.com/watch?v=4l97aNza_Zc"
    
    result = pipeline.extract_from_youtube(video_url)
    
    if result['success']:
        print(f"✓ Successfully extracted transcript from YouTube video")
        print(f"  Video ID: {result['video_id']}")
        print(f"  Segments: {result['metadata']['num_segments']}")
        print(f"  Duration: {result['metadata']['total_duration']:.1f} seconds")
        print(f"  Characters: {result['metadata']['total_characters']}")
    else:
        print(f"✗ Failed to extract transcript: {result['error']}")

def example_youtube_playlist():
    """Example: Extract transcripts from a YouTube playlist"""
    print("\n=== YouTube Playlist Extraction Example ===")
    
    pipeline = TextExtractionPipeline()
    
    # Example YouTube playlist URL
    playlist_url = "https://www.youtube.com/@MrBeast/videos"
    
    result = pipeline.extract_from_youtube(playlist_url, max_videos=2)
    
    if result['success']:
        summary = result['summary']
        print(f"✓ Successfully processed playlist")
        print(f"  Total videos: {summary['total_videos']}")
        print(f"  Processed: {summary['processed_videos']}")
        print(f"  Successful: {summary['successful_extractions']}")
        print(f"  Failed: {summary['failed_extractions']}")
        print(f"  Success rate: {summary['success_rate']:.1f}%")
    else:
        print(f"✗ Failed to process playlist: {result['error']}")

def example_mixed_inputs():
    """Example: Process mixed inputs (files and YouTube URLs)"""
    print("\n=== Mixed Inputs Example ===")
    
    pipeline = TextExtractionPipeline()
    
    # Mixed inputs
    inputs = [
        "data/input/sample.pdf",  # File
        "data/input/sample.docx",  # File
        "https://www.youtube.com/watch?v=4l97aNza_Zc",  # YouTube video
        {
            'type': 'youtube',
            'url': 'https://www.youtube.com/@MrBeast/videos'
        }  # YouTube playlist
    ]
    
    result = pipeline.process_mixed_inputs(inputs)
    
    print(f"✓ Processed mixed inputs")
    print(f"  Total inputs: {result['summary']['total_inputs']}")
    print(f"  Successful: {result['summary']['successful_extractions']}")
    print(f"  Failed: {result['summary']['failed_extractions']}")
    print(f"  Success rate: {result['summary']['success_rate']:.1f}%")

def main():
    """Run all examples"""
    print("LMS AI Assistant - Text Extraction Pipeline Examples")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # Run examples
    example_single_file()
    example_youtube_video()
    example_youtube_playlist()
    example_mixed_inputs()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("Check data/output/ for results and logs")

if __name__ == "__main__":
    main()
