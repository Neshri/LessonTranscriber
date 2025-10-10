#!/usr/bin/env python3
"""
Experimental Educational Content Detection Script

Purpose: Test different audio scenarios to distinguish educational content from personal recordings
Usage: python lecture_detector.py

Current Focus:
- Accepts: Lectures, documentaries, educational speeches, informative content
- Rejects: Personal conversations, background noise, forgotten recordings

For testing different scenarios:
1. Add audio files to lesson_audio/ folder
2. Run script to see classification results
3. Adjust thresholds in classify_as_lecture() method as needed
4. Test with real microphone recordings when available

Note: Trimming suggestions are VERY conservative and only trigger for clear
forgotten recording scenarios (3hrs+ with significant trailing noise).
"""

import os
import sys
import subprocess
import re
import json
from pathlib import Path

class LectureDetector:
    def __init__(self):
        self.audio_folder = "lesson_audio"

    def detect_speech_segments(self, audio_path):
        """
        Use ffmpeg silencedetect to find speech vs non-speech segments
        Returns list of segments with start/end times and types
        """
        try:
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'silencedetect=noise=-40dB:duration=1.0',
                '-f', 'null', '-'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                print(f"  ERROR: ffmpeg failed for {audio_path}")
                return None

            # Parse silence detection output
            segments = []
            current_pos = 0.0

            for line in result.stderr.split('\n'):
                if 'silence_start:' in line:
                    match = re.search(r'silence_start:\s*(\d+\.?\d*)', line)
                    if match:
                        end_time = float(match.group(1))
                        if end_time > current_pos:
                            segments.append({
                                'start': current_pos,
                                'end': end_time,
                                'type': 'speech',
                                'duration': end_time - current_pos
                            })
                        current_pos = end_time
                elif 'silence_end:' in line:
                    match = re.search(r'silence_end:\s*(\d+\.?\d*)', line)
                    if match:
                        current_pos = float(match.group(1))

            # Add final segment if there's remaining audio
            if current_pos < self.get_audio_duration(audio_path):
                final_duration = self.get_audio_duration(audio_path) - current_pos
                segments.append({
                    'start': current_pos,
                    'end': self.get_audio_duration(audio_path),
                    'type': 'speech',
                    'duration': final_duration
                })

            return segments

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: Timeout analyzing {audio_path}")
            return None
        except Exception as e:
            print(f"  ERROR: Error analyzing {audio_path}: {e}")
            return None

    def get_audio_duration(self, audio_path):
        """Get total duration of audio file"""
        try:
            cmd = [
                'ffprobe', '-i', audio_path,
                '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip()) if result.returncode == 0 else 0
        except:
            return 0

    def get_volume_info(self, audio_path):
        """Get volume information for audio file"""
        try:
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'volumedetect',
                '-f', 'null', '-'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                return None

            # Parse volume info
            for line in result.stderr.split('\n'):
                if 'mean_volume:' in line:
                    match = re.search(r'mean_volume:\s*(-?\d+\.?\d*)\s*dB', line)
                    if match:
                        return float(match.group(1))

            return None

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: Volume detection timeout for {audio_path}")
            return None
        except Exception as e:
            print(f"  ERROR: Volume detection error for {audio_path}: {e}")
            return None

    def detect_content_boundaries(self, audio_path, window_size_minutes=10):
        """
        Detect potential content boundaries using conservative analysis
        Only suggests trimming for clear forgotten recording scenarios
        """
        try:
            duration = self.get_audio_duration(audio_path)
            if duration == 0 or duration < 1800:  # Skip files shorter than 30 minutes
                return None

            print(f"  Content boundary analysis ({window_size_minutes}min windows):")

            segments = self.detect_speech_segments(audio_path)
            if not segments:
                print("    - No speech segments found")
                return None

            # CONSERVATIVE APPROACH: Only look for clear forgotten recording patterns
            total_speech_time = sum(s['duration'] for s in segments if s['type'] == 'speech')

            # Find the last substantial speech segment (not just longest)
            substantial_segments = [s for s in segments if s['type'] == 'speech' and s['duration'] > 300]  # 5+ minutes

            if not substantial_segments:
                print("    - No substantial speech content found")
                return None

            # Find when the last substantial content block ends
            last_substantial_end = max(s['end'] for s in substantial_segments)

            # Calculate silence after last substantial content
            silence_after_content = duration - last_substantial_end

            print(f"    - Last substantial content ends at: {last_substantial_end/60:.1f} minutes")
            print(f"    - Potential trailing content: {silence_after_content/60:.1f} minutes")

            # VERY CONSERVATIVE: Only suggest trimming if:
            # 1. File is very long (>2 hours)
            # 2. Last substantial content ended >1 hour ago
            # 3. Trailing content is >50% of total duration
            # 4. Total speech content is still >50% (it's actually educational)

            if (duration > 7200 and  # >2 hours
                silence_after_content > 3600 and  # Last content >1 hour ago
                silence_after_content > duration * 0.5 and  # Trailing >50% of file
                total_speech_time > duration * 0.5):  # But still mostly speech

                suggested_trim = last_substantial_end + 300  # 5 minute buffer
                print(f"    - CLEAR FORGOTTEN RECORDING DETECTED")
                print(f"    - SUGGESTED TRIM POINT: {suggested_trim/60:.1f} minutes")
                print(f"    - Would save: {(duration - suggested_trim)/60:.1f} minutes")

                return {
                    'content_end': last_substantial_end,
                    'trailing_duration': silence_after_content,
                    'suggested_trim': suggested_trim,
                    'confidence': 'high'
                }
            else:
                print("    - Complete educational file (no trimming needed)")
                return None

        except Exception as e:
            print(f"  ERROR: Content boundary detection failed: {e}")
            return None

    def analyze_file(self, audio_path):
        """Analyze a single audio file for lecture characteristics"""
        filename = os.path.basename(audio_path)
        print(f"\nAnalyzing: {filename}")

        # Get basic info
        duration = self.get_audio_duration(audio_path)
        if duration == 0:
            print("  ERROR: Could not determine duration")
            return False

        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

        # Get volume info
        volume_db = self.get_volume_info(audio_path)
        if volume_db is not None:
            print(f"  Volume: {volume_db:.1f} dB")
        else:
            print("  ERROR: Could not determine volume")

        # Get speech segments
        segments = self.detect_speech_segments(audio_path)
        if segments is None:
            print("  ERROR: Could not analyze segments")
            return False

        # Analyze segments
        total_speech_time = sum(s['duration'] for s in segments if s['type'] == 'speech')
        total_silence_time = duration - total_speech_time
        speech_percentage = (total_speech_time / duration) * 100

        print(f"  Speech time: {total_speech_time:.1f}s ({speech_percentage:.1f}%)")
        print(f"  Non-speech time: {total_silence_time:.1f}s ({100-speech_percentage:.1f}%)")

        # Check for content boundaries (trimming suggestions)
        boundary_info = self.detect_content_boundaries(audio_path)

        # Rule-based classification
        return self.classify_as_lecture(segments, duration, volume_db, speech_percentage)

    def classify_as_lecture(self, segments, duration, volume_db, speech_percentage):
        """
        Rule-based classification for educational content detection
        Returns True if likely educational content, False if likely personal/private
        """

        rules = []

        # Rule 1: Must have substantial speech content (educational content is speech-heavy)
        rule1 = speech_percentage > 50  # At least 50% speech for educational content
        rules.append(f"Speech > 50%: {speech_percentage:.1f}% ({rule1})")

        # Rule 2: Must have good volume (personal conversations might be quieter)
        rule2 = volume_db is not None and volume_db > -40  # Consistent with main config, suitable for speech
        volume_str = f"{volume_db:.1f}" if volume_db is not None else "N/A"
        rules.append(f"Volume > -40dB: {volume_str} ({rule2})")

        # Rule 3: Should have substantial continuous content (educational content has meaningful length)
        speech_segments = [s for s in segments if s['type'] == 'speech' and s['duration'] > 30]
        if speech_segments:
            longest_speech = max(s['duration'] for s in speech_segments)
            rule3 = longest_speech > 120  # At least 2 minutes of continuous content
            rules.append(f"Has educational-length content: {longest_speech:.1f}s max ({rule3})")
        else:
            longest_speech = 0  # Define for error reporting
            rule3 = False
            rules.append("No substantial educational content (False)")

        # Rule 4: Total duration should be appropriate for educational content
        # Educational content is typically 3+ minutes, but not endless personal recordings
        rule4 = 180 < duration < 21600  # Between 3 minutes and 6 hours (more generous for educational content)
        rules.append(f"Duration 3min-6hrs: {duration:.1f}s ({rule4})")

        # Rule 5: Not too much silence (personal recordings often have lots of dead air)
        silence_percentage = 100 - speech_percentage
        rule5 = silence_percentage < 60  # Less than 60% silence (educational content is mostly speech)
        rules.append(f"Silence < 60%: {silence_percentage:.1f}% ({rule5})")

        # Overall classification - be more permissive for educational content
        # Only require 4 out of 5 rules to pass (allows for different types of educational content)
        passed_rules = sum([rule1, rule2, rule3, rule4, rule5])
        is_lecture = passed_rules >= 4  # More permissive threshold

        print("  Rules evaluation (4/5 required):")
        for rule in rules:
            print(f"    - {rule}")

        print(f"  Classification: {'EDUCATIONAL CONTENT' if is_lecture else 'LIKELY PERSONAL/NON-EDUCATIONAL'}")

        # Add detailed analysis for testing
        if not is_lecture:
            print("  WHY THIS FILE WAS REJECTED:")
            if not rule1:
                print(f"    - Too little speech content: {speech_percentage:.1f}% (needs >50%)")
            if not rule2:
                print(f"    - Volume too low: {volume_db:.1f}dB (needs > -40dB)")
            if not rule3:
                print(f"    - No substantial educational content (longest segment: {longest_speech:.1f}s)")
            if not rule4:
                print(f"    - Duration inappropriate: {duration:.1f}s (needs 3min-6hrs)")
            if not rule5:
                print(f"    - Too much silence: {silence_percentage:.1f}% (needs <60%)")

        return is_lecture

    def test_all_files(self):
        """Test all audio files in the lesson_audio folder"""
        print("Starting Lecture Detection Test")
        print("=" * 50)

        if not os.path.exists(self.audio_folder):
            print(f"ERROR: Folder '{self.audio_folder}' not found!")
            return

        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        audio_files = []

        for ext in audio_extensions:
            pattern = f"*{ext}"
            audio_files.extend(Path(self.audio_folder).glob(pattern))

        if not audio_files:
            print(f"ERROR: No audio files found in '{self.audio_folder}'")
            return

        print(f"Found {len(audio_files)} audio files")

        results = []

        for audio_path in audio_files:
            is_lecture = self.analyze_file(str(audio_path))
            results.append((audio_path.name, is_lecture))

        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)

        lectures = [r for r in results if r[1]]
        non_lectures = [r for r in results if not r[1]]

        print(f"Educational Content: {len(lectures)}")
        for name, _ in lectures:
            print(f"   • {name}")

        print(f"Non-educational: {len(non_lectures)}")
        for name, _ in non_lectures:
            print(f"   • {name}")

        print(f"\nTotal: {len(results)} files analyzed")

        # Add threshold analysis
        self.analyze_detection_thresholds()

    def analyze_detection_thresholds(self):
        """Analyze what types of files would pass/fail with different thresholds"""
        print("\n" + "=" * 60)
        print("THRESHOLD ANALYSIS")
        print("=" * 60)

        print("Current thresholds:")
        print("  - Speech content: >50%")
        print("  - Volume: > -40dB")
        print("  - Content length: >2 minutes")
        print("  - Duration: 3min-6hrs")
        print("  - Silence: <60%")

        print("\nSuggested test scenarios to try:")
        print("  1. Very quiet educational content (-45dB)")
        print("  2. Short educational snippet (2 minutes)")
        print("  3. Very long forgotten recording (3hrs+ with trailing noise)")
        print("  4. Recording with mixed speech/silence (40% speech)")
        print("  5. Personal conversation (should be rejected)")
        print("  6. Background noise only (should be rejected)")

        print("\nTo modify thresholds, edit classify_as_lecture() method")
        print("To test specific thresholds, run with different audio files")


def main():
    detector = LectureDetector()
    detector.test_all_files()


if __name__ == "__main__":
    main()