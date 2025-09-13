"""
YouTube Transcript Extraction Pipeline
Extracts transcripts from YouTube videos and playlists
"""

import re
import subprocess
import json
import logging
from typing import List, Dict, Optional, Union
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

class YouTubeExtractor:
    """Extract transcripts from YouTube videos and playlists"""
    
    def __init__(self):
        self.api = YouTubeTranscriptApi()
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_video_transcript(self, video_id: str, languages: List[str] = ['en']) -> Dict[str, any]:
        """
        Extract transcript from a single YouTube video
        
        Args:
            video_id: YouTube video ID
            languages: List of language codes to try
            
        Returns:
            Dictionary containing transcript data and metadata
        """
        try:
            logger.info(f"Extracting transcript for video: {video_id}")
            
            # Get available transcripts
            transcript_list = self.api.list(video_id)
            available_languages = []
            
            for transcript in transcript_list:
                available_languages.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_manual': not transcript.is_generated
                })
            
            # Try to fetch transcript
            transcript = self.api.fetch(video_id, languages=languages)
            
            # Format transcript data
            formatted_segments = []
            full_text = ""
            
            for entry in transcript:
                segment = {
                    'start_time': entry.start,
                    'duration': entry.duration,
                    'text': entry.text,
                    'end_time': entry.start + entry.duration
                }
                formatted_segments.append(segment)
                full_text += entry.text + " "
            
            result = {
                'success': True,
                'video_id': video_id,
                'metadata': {
                    'available_languages': available_languages,
                    'selected_language': languages[0] if languages else 'en',
                    'num_segments': len(formatted_segments),
                    'total_duration': formatted_segments[-1]['end_time'] if formatted_segments else 0,
                    'total_characters': len(full_text.strip())
                },
                'segments': formatted_segments,
                'full_text': full_text.strip(),
                'extraction_method': 'youtube-transcript-api'
            }
            
            logger.info(f"Successfully extracted transcript with {len(formatted_segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting transcript for video {video_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_id': video_id,
                'extraction_method': 'youtube-transcript-api'
            }
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, any]]:
        """
        Get video information from a YouTube playlist
        
        Args:
            playlist_url: YouTube playlist URL
            
        Returns:
            List of video information dictionaries
        """
        try:
            logger.info(f"Fetching playlist information from: {playlist_url}")
            
            cmd = [
                'yt-dlp',
                '--flat-playlist',
                '--dump-json',
                '--no-warnings',
                playlist_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            videos = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        video_data = json.loads(line)
                        video_info = {
                            'id': video_data.get('id', ''),
                            'title': video_data.get('title', 'Unknown Title'),
                            'url': f"https://www.youtube.com/watch?v={video_data.get('id', '')}",
                            'duration': video_data.get('duration', 0),
                            'uploader': video_data.get('uploader', 'Unknown'),
                            'view_count': video_data.get('view_count', 0),
                            'upload_date': video_data.get('upload_date', '')
                        }
                        videos.append(video_info)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Found {len(videos)} videos in playlist")
            return videos
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error fetching playlist: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching playlist: {e}")
            return []
    
    def extract_playlist_transcripts(self, playlist_url: str, languages: List[str] = ['en'], 
                                   max_videos: Optional[int] = None) -> Dict[str, any]:
        """
        Extract transcripts from all videos in a playlist
        
        Args:
            playlist_url: YouTube playlist URL
            languages: List of language codes to try
            max_videos: Maximum number of videos to process
            
        Returns:
            Dictionary containing all transcript data and summary
        """
        # Get playlist videos
        videos = self.get_playlist_videos(playlist_url)
        
        if not videos:
            return {
                'success': False,
                'error': 'No videos found in playlist',
                'playlist_url': playlist_url
            }
        
        # Limit videos if specified
        if max_videos and max_videos < len(videos):
            videos = videos[:max_videos]
            logger.info(f"Processing first {max_videos} videos only")
        
        # Process each video
        results = []
        successful_extractions = 0
        failed_extractions = 0
        
        for i, video in enumerate(videos, 1):
            video_id = video['id']
            video_title = video['title']
            
            logger.info(f"Processing video {i}/{len(videos)}: {video_title}")
            
            transcript_result = self.get_video_transcript(video_id, languages)
            transcript_result['video_info'] = video
            transcript_result['playlist_position'] = i
            
            results.append(transcript_result)
            
            if transcript_result['success']:
                successful_extractions += 1
            else:
                failed_extractions += 1
        
        # Create summary
        summary = {
            'playlist_url': playlist_url,
            'total_videos': len(videos),
            'processed_videos': len(results),
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': (successful_extractions / len(videos)) * 100 if videos else 0
        }
        
        return {
            'success': True,
            'summary': summary,
            'videos': results,
            'extraction_method': 'youtube-transcript-api + yt-dlp'
        }
    
    def extract_from_url(self, url: str, languages: List[str] = ['en']) -> Dict[str, any]:
        """
        Extract transcript from a YouTube URL (video or playlist)
        
        Args:
            url: YouTube video or playlist URL
            languages: List of language codes to try
            
        Returns:
            Dictionary containing transcript data
        """
        # Check if it's a playlist
        if 'playlist' in url.lower():
            return self.extract_playlist_transcripts(url, languages)
        
        # Extract video ID and process single video
        video_id = self.extract_video_id(url)
        if not video_id:
            return {
                'success': False,
                'error': 'Could not extract video ID from URL',
                'url': url
            }
        
        return self.get_video_transcript(video_id, languages)
