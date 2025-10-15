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

Note: Trimming suggestions are MODERATELY sensitive for testing purposes.
Triggers for forgotten recording scenarios (15min+ with 5min+ trailing silence).
This helps test spliced files and moderate forgotten recording scenarios.
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
                '-af', 'silencedetect=noise=-35dB:duration=2.0',
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

    def normalize_audio_volume(self, audio_path, target_volume_db=-20.0, output_folder="output"):
        """
        Normalize audio volume to target level and save to output folder
        Returns path to normalized file if successful, None if failed or no normalization needed
        """
        import tempfile
        import shutil

        try:
            # Check current volume
            current_volume = self.get_volume_info(audio_path)
            if current_volume is None:
                print(f"  ERROR: Cannot determine volume for {audio_path}")
                return None

            # Skip normalization if already within acceptable range
            volume_diff = abs(current_volume - target_volume_db)
            if volume_diff < 2.0:  # Within 2dB of target, no need to normalize
                print(f"  SKIP: Volume {current_volume:.1f}dB already close to target {target_volume_db}dB")
                return None

            print(f"  NORMALIZING: {current_volume:.1f}dB to {target_volume_db}dB")

            # Create output folder if it doesn't exist
            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filename - use .m4a extension for AAC
            base_name = Path(audio_path).stem
            output_filename = f"{base_name}_normalized.m4a"
            output_path = output_dir / output_filename

            # Use temporary file during processing
            with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Calculate volume adjustment needed
                volume_adjust_db = target_volume_db - current_volume

                # Normalize using volume filter with re-encoding for compatibility
                volume_adjust_db = target_volume_db - current_volume
                volume_multiplier = 10 ** (volume_adjust_db / 20.0)

                cmd = [
                    'ffmpeg', '-i', audio_path,
                    '-af', f'volume={volume_multiplier}',
                    '-c:a', 'aac',  # Re-encode to AAC for compatibility
                    '-b:a', '128k',  # Set bitrate
                    '-y', temp_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5min timeout

                if result.returncode == 0:
                    # Move temp file to final location
                    shutil.move(temp_path, output_path)
                    print(f"  SUCCESS: Normalized audio saved to {output_path}")
                    return str(output_path)
                else:
                    print(f"  ERROR: ffmpeg normalization failed")
                    print(f"  Command: {' '.join(cmd)}")
                    print(f"  Return code: {result.returncode}")
                    if result.stderr:
                        print(f"  stderr: {result.stderr[-500:]}")  # Last 500 chars
                    return None

            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT: Volume normalization timed out for {audio_path}")
                return None
            except Exception as e:
                print(f"  ERROR: Volume normalization failed: {e}")
                return None
            finally:
                # Clean up temp file if it still exists
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            print(f"  ERROR: Volume normalization setup failed: {e}")
            return None

    def detect_content_boundaries(self, audio_path, window_size_minutes=10):
        """
        Detect content boundaries using gap analysis between speech segments
        For spliced files: looks for significant gaps that indicate splice points
        """
        try:
            duration = self.get_audio_duration(audio_path)
            if duration == 0 or duration < 600:  # Skip files shorter than 10 minutes
                return None

            print(f"  Content boundary analysis (GAP DETECTION):")

            segments = self.detect_speech_segments(audio_path)
            if not segments:
                print("    - No speech segments found")
                return None

            # NEW APPROACH: Find gaps between speech segments for spliced file detection
            total_speech_time = sum(s['duration'] for s in segments if s['type'] == 'speech')

            # Get only substantial speech segments (>30 seconds)
            substantial_segments = [s for s in segments if s['type'] == 'speech' and s['duration'] > 30]

            if len(substantial_segments) <= 1:
                # Single or no substantial segments - check if file is much longer than content
                if substantial_segments:
                    single_segment = substantial_segments[0]
                    segment_end = single_segment['end']
                    if duration > segment_end + 180:  # File >3min longer than single content block
                        suggested_trim = segment_end + 60
                        print(f"    - SINGLE CONTENT BLOCK WITH LONG TRAILING")
                        print(f"    - Content ends at: {segment_end/60:.1f} minutes")
                        print(f"    - SUGGESTED TRIM POINT: {suggested_trim/60:.1f} minutes")
                        print(f"    - Would save: {(duration - suggested_trim)/60:.1f} minutes")

                        return {
                            'content_end': segment_end,
                            'trailing_duration': duration - segment_end,
                            'suggested_trim': suggested_trim,
                            'confidence': 'high'
                        }
                print("    - No clear content boundaries detected")
                return None

            # Multiple substantial segments - look for largest gap (likely splice point)
            substantial_segments.sort(key=lambda x: x['start'])

            max_gap = 0
            gap_position = 0

            for i in range(len(substantial_segments) - 1):
                gap_start = substantial_segments[i]['end']
                gap_end = substantial_segments[i + 1]['start']
                gap_duration = gap_end - gap_start

                if gap_duration > max_gap:
                    max_gap = gap_duration
                    gap_position = gap_end

            print(f"    - Largest gap between speech: {max_gap/60:.1f} minutes")
            print(f"    - Gap ends at: {gap_position/60:.1f} minutes")

            # Suggest trimming if there's a significant gap
            if max_gap > 120:  # Gap > 2 minutes
                suggested_trim = gap_position + 60  # Add buffer after gap
                print(f"    - SPLICE POINT DETECTED")
                print(f"    - SUGGESTED TRIM POINT: {suggested_trim/60:.1f} minutes")
                print(f"    - Would save: {(duration - suggested_trim)/60:.1f} minutes")

                return {
                    'splice_point': gap_position,
                    'gap_duration': max_gap,
                    'suggested_trim': suggested_trim,
                    'confidence': 'high'
                }

            print("    - No significant gaps detected")
            return None

        except Exception as e:
            print(f"  ERROR: Content boundary detection failed: {e}")
            return None

    def analyze_file(self, audio_path, normalize_volume=False, output_folder="output"):
        """Analyze a single audio file for lecture characteristics

        Args:
            audio_path: Path to audio file to analyze
            normalize_volume: If True, normalize volume and return normalized file path
            output_folder: Folder to save normalized files

        Returns:
            If normalize_volume=False: Boolean indicating if file is educational content
            If normalize_volume=True: Tuple of (is_educational, normalized_file_path)
        """
        filename = os.path.basename(audio_path)
        print(f"\nAnalyzing: {filename}")

        # Get basic info
        duration = self.get_audio_duration(audio_path)
        if duration == 0:
            print("  ERROR: Could not determine duration")
            if normalize_volume:
                return False, None
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
            if normalize_volume:
                return False, None
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
        is_lecture = self.classify_as_lecture(segments, duration, volume_db, speech_percentage)

        # Optional volume normalization
        normalized_path = None
        if normalize_volume and is_lecture:
            # Only normalize educational content
            normalized_path = self.normalize_audio_volume(audio_path, output_folder=output_folder)
            if normalized_path:
                print(f"  VOLUME NORMALIZED: {normalized_path}")

        if normalize_volume:
            return is_lecture, normalized_path
        return is_lecture

    def process_and_normalize_file(self, audio_path, output_folder="output"):
        """Process file and create normalized version if it's educational content

        Returns:
            Tuple of (is_educational, normalized_path_or_none)
        """
        return self.analyze_file(audio_path, normalize_volume=True, output_folder=output_folder)

    def classify_as_lecture(self, segments, duration, volume_db, speech_percentage):
        """
        Rule-based classification for educational content detection
        Returns True if likely educational content, False if likely personal/private
        """

        rules = []

        # Rule 1: Must have substantial speech content (educational content is speech-heavy)
        rule1 = speech_percentage > 30  # At least 30% speech for educational content (more lenient for testing)
        rules.append(f"Speech > 30%: {speech_percentage:.1f}% ({rule1})")

        # Rule 2: Must have good volume (personal conversations might be quieter)
        rule2 = volume_db is not None and volume_db > -40  # Consistent with main config, suitable for speech
        volume_str = f"{volume_db:.1f}" if volume_db is not None else "N/A"
        rules.append(f"Volume > -40dB: {volume_str} ({rule2})")

        # Rule 3: Should have substantial continuous content (educational content has meaningful length)
        speech_segments = [s for s in segments if s['type'] == 'speech' and s['duration'] > 60]
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
                print(f"    - Too little speech content: {speech_percentage:.1f}% (needs >30%)")
            if not rule2:
                print(f"    - Volume too low: {volume_db:.1f}dB (needs > -40dB)")
            if not rule3:
                print(f"    - No substantial educational content (longest segment: {longest_speech:.1f}s)")
            if not rule4:
                print(f"    - Duration inappropriate: {duration:.1f}s (needs 3min-6hrs)")
            if not rule5:
                print(f"    - Too much silence: {silence_percentage:.1f}% (needs <60%)")

        return is_lecture

    def test_all_files(self, normalize_volume=False, output_folder="output"):
        """Test all audio files in the lesson_audio folder

        Args:
            normalize_volume: If True, normalize volume of educational content
            output_folder: Folder to save normalized files
        """
        print("Starting Lecture Detection Test")
        if normalize_volume:
            print("WITH VOLUME NORMALIZATION ENABLED")
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
            if normalize_volume:
                is_lecture, normalized_path = self.process_and_normalize_file(str(audio_path), output_folder=output_folder)
                results.append((audio_path.name, is_lecture, normalized_path))
            else:
                is_lecture = self.analyze_file(str(audio_path))
                results.append((audio_path.name, is_lecture, None))

        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)

        lectures = [r for r in results if r[1]]
        non_lectures = [r for r in results if not r[1]]

        print(f"Educational Content: {len(lectures)}")
        for name, _, normalized_path in lectures:
            status = ""
            if normalized_path:
                status = " (NORMALIZED)"
            print(f"   • {name}{status}")

        print(f"Non-educational: {len(non_lectures)}")
        for name, _, _ in non_lectures:
            print(f"   • {name}")

        print(f"\nTotal: {len(results)} files analyzed")

        if normalize_volume:
            normalized_count = sum(1 for r in results if r[2] is not None)
            print(f"Files normalized: {normalized_count}")

        # Add threshold analysis
        self.analyze_detection_thresholds()

    def analyze_detection_thresholds(self):
        """Analyze what types of files would pass/fail with different thresholds"""
        print("\n" + "=" * 60)
        print("THRESHOLD ANALYSIS")
        print("=" * 60)

        print("Current thresholds:")
        print("  - Speech content: >30%")
        print("  - Volume: > -40dB")
        print("  - Content length: >2 minutes")
        print("  - Duration: 3min-6hrs")
        print("  - Silence: <60%")

        print("\nSuggested test scenarios to try:")
        print("  1. Very quiet educational content (-45dB)")
        print("  2. Short educational snippet (2 minutes)")
        print("  3. Test forgotten recording (8min+ with 1min+ trailing silence)")
        print("  4. Recording with mixed speech/silence (40% speech)")
        print("  5. Personal conversation (should be rejected)")
        print("  6. Background noise only (should be rejected)")
 
        print("\nTo modify thresholds, edit classify_as_lecture() method")
        print("To test specific thresholds, run with different audio files")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Lecture Detection with Optional Volume Normalization')
    parser.add_argument('--normalize', action='store_true',
                       help='Enable volume normalization for educational content')
    parser.add_argument('--output', default='output',
                       help='Output folder for normalized files (default: output)')

    # Parse known args to allow running without arguments
    args, unknown = parser.parse_known_args()

    detector = LectureDetector()
    detector.test_all_files(normalize_volume=args.normalize, output_folder=args.output)


if __name__ == "__main__":
    main()