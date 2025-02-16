from pydub import AudioSegment
import os

def merge_audio(instrumental_path, vocal_path, output_path):
    """Merge instrumental and vocal tracks"""
    try:
        # Load both audio files
        instrumental = AudioSegment.from_wav(instrumental_path)
        vocal = AudioSegment.from_wav(vocal_path)
        
        # Overlay tracks (mix them together)
        combined = instrumental.overlay(vocal)
        
        # Export merged audio
        combined.export(output_path, format='wav')
        print(f"Successfully merged audio to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error merging audio: {str(e)}")
        return None