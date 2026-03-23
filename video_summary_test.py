import logging
from openscenesense_ollama.models import AnalysisPrompts
from openscenesense_ollama.transcriber import WhisperTranscriber
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import UniformFrameSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# KB-Whisper large with strict revision for most verbatim transcription
transcriber = WhisperTranscriber(
    model_name="KBLab/kb-whisper-large",
    device="cuda"
)

custom_prompts = AnalysisPrompts(
    frame_analysis=(
        "You are analyzing a screen capture from an IT lesson. "
        "Describe only what is explicitly visible: open windows, application names, "
        "terminal output, file/folder structures, dialog boxes, and any text on screen. "
        "Do not infer or assume anything not directly visible."
    ),
    detailed_summary=(
        "You are summarizing an IT lesson from a screen recording. "
        "Based strictly on the visual timeline and audio transcript below, "
        "write a faithful summary of what was taught. "
        "Do not add information not present in the source material.\n\n"
        "Video duration: {duration:.1f} seconds\n\n"
        "Screen timeline:\n{timeline}\n\n"
        "Audio transcript:\n{transcript}"
    ),
    brief_summary=(
        "Based strictly on this {duration:.1f}-second IT lesson recording, "
        "provide a concise summary of what was demonstrated and explained. "
        "Only include what is explicitly present in the timeline and transcript.\n\n"
        "Screen timeline:\n{timeline}\n\n"
        "Transcript:\n{transcript}"
    )
)

analyzer = OllamaVideoAnalyzer(
    frame_analysis_model="qwen3-vl:32b",
    summary_model="qwen3:32b",
    min_frames=5,
    max_frames=15,
    frames_per_minute=3.0,
    frame_selector=UniformFrameSelector(),
    audio_transcriber=transcriber,
    prompts=custom_prompts,
    request_timeout=600.0,   # 10 min per request — qwen3-vl:32b is slow
    request_retries=1,
    log_level=logging.INFO
)

video_path = "windows7kontrollpanel.mp4"
try:
    results = analyzer.analyze_video(video_path)
except Exception as exc:
    logging.getLogger(__name__).error("analyze_video raised an exception: %s", exc, exc_info=True)
    raise

output_file = "summary_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Brief Summary:\n")
    f.write(results.get('brief_summary', '(missing)'))
    f.write("\n\nDetailed Summary:\n")
    f.write(results.get('summary', '(missing)'))

print(f"\nResults have been written to {output_file}")