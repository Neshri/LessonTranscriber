
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
