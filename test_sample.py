#!/usr/bin/env python3
"""
Test script for Lesson Transcriber - demonstrates usage without audio file
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    import requests
except ImportError:
    print("ERROR: requests not installed.")
    sys.exit(1)

def test_ollama_connection(ollama_url="http://localhost:11434"):
    """Test connection to Ollama"""
    try:
        print(f"Testing Ollama connection at {ollama_url}...")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print("SUCCESS: Ollama is running and has models:")
                for model in models:
                    print(f"   - {model.get('name', 'unknown')}")
                return True
            else:
                print("WARNING: Ollama is running but no models found.")
                print("   Run: ollama pull gemma3:4b-it-qat")
                return False
        else:
            print(f"ERROR: Ollama API returned {response.status_code}")
            return False

    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is installed and running.")
        print("   Download: https://ollama.com/")
        print("   Start: ollama serve")
        return False

def test_whisper_import():
    """Test Whisper import"""
    try:
        print("Testing Whisper import...")
        import whisper
        print("SUCCESS: Whisper imported successfully")
        return True
    except ImportError:
        print("ERROR: Whisper not installed.")
        print("   Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"ERROR: Whisper import failed: {e}")
        return False

def create_sample_readme():
    """Create sample usage instructions"""
    sample_readme = """
# How to Test Lesson Transcriber

## 1. Prepare Audio Files
Place audio files (mp3, wav, m4a, flac, ogg) in the 'lesson_audio/' subdirectory or any directory you specify.
For example: lesson_audio/lesson.mp3

## 2. Ensure Prerequisites
- Python virtual environment activated
- Ollama running and gemma3:4b-it-qat model pulled (`ollama pull gemma3:4b-it-qat`)
- Dependencies installed

## 3. Run the Transcriber
### Options:
- `python main.py` (defaults to processing all audio files in 'lesson_audio/' directory)
- `python main.py lesson_audio/lesson.mp3` (processes single file)
- `python main.py ./lesson_audio/` (processes all files in specified directory)
- `python main.py /path/to/your/lesson_audio/lesson.mp3` (absolute path to file)

## 4. Expected Output
- Files saved to 'output/' directory:
  - [filename]_transcript.txt
  - [filename]_summary.txt
- Console output with full transcript and summary for each processed file

## Troubleshooting
- If Ollama connection fails: `ollama serve`
- If Whisper fails: Check virtual environment and dependencies
- If audio file issues: Ensure supported format and file exists
"""
    Path("SAMPLE_USAGE.md").write_text(sample_readme, encoding='utf-8')
    print("Created SAMPLE_USAGE.md with testing instructions")

def main():
    print("="*50)
    print("LESSON TRANSCRIBER TEST SUITE")
    print("="*50)
    print()

    # Test components
    ollama_ok = test_ollama_connection()
    whisper_ok = test_whisper_import()

    print()
    print("-" * 50)
    print("TEST RESULTS:")
    print("-" * 50)
    print(f"Ollama Connection: {'PASS' if ollama_ok else 'FAIL'}")
    print(f"Whisper Import:    {'PASS' if whisper_ok else 'FAIL'}")

    if ollama_ok and whisper_ok:
        print("\nSUCCESS: All tests passed! Ready to transcribe lessons.")
        print("\nNext steps:")
        print("1. Place audio file in 'lesson_audio/' directory (mp3/wav/m4a/flac/ogg)")
        print("2. Run: python main.py [audio_source] (defaults to 'lesson_audio/')")
        print("   Examples:")
        print("   - python main.py (uses 'lesson_audio/' directory)")
        print("   - python main.py my_file.mp3")
        print("   - python main.py ./lesson_audio/")
        create_sample_readme()
    else:
        print("\nWARNING: Some tests failed. Please fix issues above before using.")
        print("\nMissing dependencies:")
        if not ollama_ok:
            print("- Ollama not running or no models installed")
        if not whisper_ok:
            print("- Whisper not installed")

        print("\nTo fix:")
        print("1. Install Ollama: https://ollama.com/")
        print("2. Pull model: ollama pull gemma3:4b-it-qat")
        print("3. Start Ollama: ollama serve")
        print("4. Install deps: pip install -r requirements.txt")

if __name__ == "__main__":
    main()