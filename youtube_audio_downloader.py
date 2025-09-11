#!/usr/bin/env python3
"""
YouTube Audio Downloader using yt-dlp
"""

import sys
import yt_dlp
from pathlib import Path

def download_audio_from_youtube(url, output_dir="lesson_audio"):
    """
    Downloads the best quality audio from a YouTube URL and saves it as an MP3
    in the specified directory.
    """
    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options for high-quality MP3 download
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',  # Bitrate in kbit/s
        }],
        # Save files to the 'lesson_audio' directory with their title
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False, # Set to True to hide console output
    }

    print(f"\nAttempting to download audio from: {url}")
    print(f"File will be saved in: {output_path.resolve()}")

    try:
        # Create a YoutubeDL object and download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("\n✅ Download and conversion successful!")

    except yt_dlp.utils.DownloadError as e:
        print(f"\n❌ Error: Failed to download the video. Reason: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python download_youtube.py \"<YOUTUBE_URL>\"")
        sys.exit(1)
        
    video_url = sys.argv[1]
    download_audio_from_youtube(video_url)