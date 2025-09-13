#!/usr/bin/env python3
"""
LMS AI Assistant - Text Extraction Pipeline
Main entry point for the text extraction system
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipelines.text_extraction_pipeline import TextExtractionPipeline

# Auto-load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=True)
except Exception:
    pass

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='LMS AI Assistant - Text Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a single PDF file
  python main.py --file document.pdf
  
  # Extract from a directory of documents
  python main.py --directory ./documents/
  
  # Extract from a YouTube video
  python main.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Extract from a YouTube playlist
  python main.py --youtube "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-videos 5
  
  # Extract from multiple sources
  python main.py --file doc1.pdf --file doc2.docx --youtube "https://youtube.com/watch?v=VIDEO_ID"
        """
    )
    
    # Input options
    parser.add_argument('--file', action='append', help='Path to a file to process (PDF, DOC, DOCX)')
    parser.add_argument('--directory', help='Path to directory containing files to process')
    parser.add_argument('--youtube', action='append', help='YouTube video or playlist URL')
    
    # Processing options
    parser.add_argument('--languages', nargs='+', default=['en'], 
                       help='Language codes for YouTube transcripts (default: en)')
    parser.add_argument('--max-videos', type=int, 
                       help='Maximum number of videos to process from playlists')
    parser.add_argument('--output-dir', default='data/output',
                       help='Output directory for results (default: data/output)')
    
    # Output options
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to JSON file')
    parser.add_argument('--output-filename', 
                       help='Custom filename for saved results')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--list-formats', action='store_true',
                       help='List supported file formats and exit')
    
    # Vector store options
    parser.add_argument('--ingest-results', action='store_true',
                        help='Ingest extracted results into local FAISS store')
    parser.add_argument('--query', help='Run a semantic query against the local FAISS store')
    parser.add_argument('--ask', help='RAG answer using vector store + LLM (requires GROQ_API_KEY)')
    parser.add_argument('--top-k', type=int, default=5, help='Top K results for semantic search')
    parser.add_argument('--embed-model', default='all-MiniLM-L6-v2',
                        help='Sentence-transformers model name for embeddings')

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # List supported formats if requested
    if args.list_formats:
        pipeline = TextExtractionPipeline()
        formats = pipeline.get_supported_formats()
        print("Supported formats:")
        for extractor_type, extensions in formats.items():
            print(f"  {extractor_type}: {', '.join(extensions)}")
        return
    
    # Validate inputs
    if not any([args.file, args.directory, args.youtube]):
        parser.error("At least one input source (--file, --directory, or --youtube) must be specified")
    
    # Initialize pipeline
    try:
        pipeline = TextExtractionPipeline(output_dir=args.output_dir)
        logger.info("Text extraction pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    # Process inputs
    all_results = []
    
    # Process files
    if args.file:
        for file_path in args.file:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue
            
            logger.info(f"Processing file: {file_path}")
            result = pipeline.extract_from_file(file_path)
            all_results.append({
                'type': 'file',
                'input': file_path,
                'result': result
            })
    
    # Process directory
    if args.directory:
        if not os.path.exists(args.directory):
            logger.error(f"Directory not found: {args.directory}")
        else:
            logger.info(f"Processing directory: {args.directory}")
            result = pipeline.extract_from_directory(args.directory)
            all_results.append({
                'type': 'directory',
                'input': args.directory,
                'result': result
            })
    
    # Process YouTube URLs
    if args.youtube:
        for url in args.youtube:
            logger.info(f"Processing YouTube URL: {url}")
            result = pipeline.extract_from_youtube(
                url, 
                languages=args.languages,
                max_videos=args.max_videos
            )
            all_results.append({
                'type': 'youtube',
                'input': url,
                'result': result
            })
    
    # Print results summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    total_successful = 0
    total_failed = 0
    
    for item in all_results:
        result = item['result']
        input_name = item['input']
        
        if result.get('success', False):
            if item['type'] == 'directory':
                summary = result.get('summary', {})
                successful = summary.get('successful_extractions', 0)
                failed = summary.get('failed_extractions', 0)
                print(f"✓ {item['type'].title()}: {input_name}")
                print(f"  Successful: {successful}, Failed: {failed}")
                total_successful += successful
                total_failed += failed
            elif item['type'] == 'youtube':
                if 'videos' in result:
                    # Playlist
                    summary = result.get('summary', {})
                    successful = summary.get('successful_extractions', 0)
                    failed = summary.get('failed_extractions', 0)
                    print(f"✓ {item['type'].title()}: {input_name}")
                    print(f"  Videos processed: {successful}, Failed: {failed}")
                    total_successful += successful
                    total_failed += failed
                else:
                    # Single video
                    print(f"✓ {item['type'].title()}: {input_name}")
                    total_successful += 1
            else:
                # Single file
                print(f"✓ {item['type'].title()}: {input_name}")
                total_successful += 1
        else:
            print(f"✗ {item['type'].title()}: {input_name}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            total_failed += 1
    
    print(f"\nTotal successful extractions: {total_successful}")
    print(f"Total failed extractions: {total_failed}")
    print(f"Success rate: {(total_successful/(total_successful+total_failed)*100):.1f}%" if (total_successful+total_failed) > 0 else "N/A")
    
    # Optional: Ingest into vector store
    if args.ingest_results:
        from utils.ingest import ingest_documents
        flat_docs: List[Dict[str, Any]] = []
        for item in all_results:
            res = item['result']
            if isinstance(res, dict) and res.get('success'):
                flat_docs.append(res)
            elif isinstance(res, dict) and res.get('videos'):
                # Playlist: append each
                for v in res['videos']:
                    if v.get('success'):
                        flat_docs.append(v)
        store_dir = ingest_documents(flat_docs, model_name=args.embed_model)
        print(f"\nIngested {len(flat_docs)} documents into vector store at: {store_dir}")

    # Optional: Query vector store
    if args.query:
        from utils.ingest import semantic_search
        hits = semantic_search(args.query, top_k=args.top_k, model_name=args.embed_model)
        print("\nSemantic Search Results:")
        for score, meta in hits:
            print(f"  score={score:.3f} source={meta.get('source')} chunk={meta.get('chunk_index')} chars={meta.get('char_count')}")

    # Optional: Retrieval-Augmented Generation answer
    if args.ask:
        # Use MMR retriever for better diversity
        try:
            from utils.retrieval import mmr_retrieve
            contexts = mmr_retrieve(args.ask, top_k=args.top_k, model_name=args.embed_model)
        except Exception:
            from utils.ingest import semantic_search
            hits = semantic_search(args.ask, top_k=args.top_k, model_name=args.embed_model)
            contexts = [meta for _, meta in hits]
        if not contexts:
            print("\nNo context found in vector store. Try ingesting first with --ingest-results.")
        else:
            from utils.rag_llm import answer_with_context
            answer = answer_with_context(args.ask, contexts)
            print("\nAnswer:\n" + answer)

    # Save results if requested
    if args.save_results:
        try:
            output_file = pipeline.save_results(all_results, args.output_filename)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return 1
    
    print(f"\nOutput directory: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
