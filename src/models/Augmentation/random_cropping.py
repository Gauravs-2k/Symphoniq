#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random cropping augmentation for audio files
"""

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union

def crop_audio_segment(audio: np.ndarray, sr: int = 32000, 
                       segment_duration: float = 5.0,
                       normalize: bool = True) -> tuple:
    """
    Randomly crop a segment from audio
    
    Args:
        audio: Audio array
        sr: Sample rate
        segment_duration: Duration of segment in seconds
        normalize: Whether to normalize audio after cropping
        
    Returns:
        Tuple of (cropped_audio, start_time, end_time)
    """
    # Calculate segment length in samples
    segment_length = int(segment_duration * sr)
    
    # Check if audio is long enough to crop
    if len(audio) <= segment_length:
        # Audio is too short, return the whole thing
        return audio, 0.0, len(audio) / sr
    
    # Choose a random start point
    max_start_idx = len(audio) - segment_length
    start_idx = np.random.randint(0, max_start_idx)
    end_idx = start_idx + segment_length
    
    # Crop the audio
    cropped_audio = audio[start_idx:end_idx].copy()
    
    # Calculate times in seconds
    start_time = start_idx / sr
    end_time = end_idx / sr
    
    # Normalize if requested
    if normalize:
        cropped_audio = librosa.util.normalize(cropped_audio)
        
    return cropped_audio, start_time, end_time


def create_cropped_prompt(piece_name: str, instrument: str, start_time: float, end_time: float) -> str:
    """
    Create a text prompt for cropped audio examples
    
    Args:
        piece_name: Name of the piece
        instrument: Instrument name
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Text prompt describing the cropped segment
    """
    # Format times nicely
    start_str = f"{int(start_time // 60)}:{int(start_time % 60):02d}" if start_time >= 60 else f"{start_time:.1f}s"
    end_str = f"{int(end_time // 60)}:{int(end_time % 60):02d}" if end_time >= 60 else f"{end_time:.1f}s"
    
    prompt = (
        f"Generate a solo {instrument} performance playing {piece_name}. "
        f"This is a segment from {start_str} to {end_str}. "
        f"The {instrument} should play with proper breath control, fluid phrasing, "
        f"clear articulation, and accurate timing. This should sound like a professional "
        f"{instrument} recording with natural dynamics and expressive interpretation."
    )
    
    return prompt


def create_cropped_examples(
    audio_files: List[Path], 
    instrument: str,
    num_crops_per_file: int = 3,
    segment_durations: List[float] = [3.0, 5.0, 7.0],
    preprocess_audio_file = None,
    desc: str = None
) -> List[Dict[str, Any]]:
    """
    Create randomly cropped versions of audio files
    
    Args:
        audio_files: List of audio file paths
        instrument: Instrument name for prompts
        num_crops_per_file: Number of crops to extract per file
        segment_durations: List of segment durations to sample from
        preprocess_audio_file: Function to preprocess audio file
        desc: Description for progress bar
        
    Returns:
        List of examples with cropped audio segments
    """
    examples = []
    
    # Create description for progress bar if not provided
    if desc is None:
        desc = f"Creating randomly cropped versions"
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc=f"Cropping audio segments"):
        try:
            # Get base name for prompt
            piece_name = audio_path.stem.replace('_', ' ')
            
            # Load original audio
            audio = preprocess_audio_file(audio_path)
            
            # Skip if audio is too short (less than min duration)
            min_duration = min(segment_durations) if segment_durations else 3.0
            if len(audio) < min_duration * 32000:  # Assuming 32kHz
                print(f"Skipping {audio_path}: too short for cropping")
                continue
            
            # Create multiple crops
            for i in range(num_crops_per_file):
                # Randomly select segment duration
                segment_duration = np.random.choice(segment_durations)
                
                # Crop the audio
                cropped, start_time, end_time = crop_audio_segment(
                    audio, 
                    segment_duration=segment_duration
                )
                
                # Create prompt
                prompt = create_cropped_prompt(piece_name, instrument, start_time, end_time)
                
                # Add to examples
                examples.append({
                    "prompt": prompt,
                    "audio": cropped.tolist(),  # Convert to list for serialization
                    "audio_path": str(audio_path),
                    "augmentation": f"crop_{i+1}_{start_time:.1f}s_to_{end_time:.1f}s"
                })
                
        except Exception as e:
            print(f"Error processing {audio_path} for random cropping: {str(e)}")
    
    return examples


# Stand-alone function to batch process a directory
def batch_create_cropped_audio(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path],
    num_crops_per_file: int = 3,
    segment_durations: List[float] = [3.0, 5.0, 7.0],
    sample_rate: int = 32000
) -> None:
    """
    Utility function to batch process a directory of audio files and save cropped segments
    
    Args:
        input_dir: Directory containing source audio files
        output_dir: Directory to save cropped audio files
        num_crops_per_file: Number of crops to extract per file
        segment_durations: List of segment durations to sample from
        sample_rate: Sample rate for saved audio
    """
    import os
    import scipy.io.wavfile
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all audio files from the input directory
    audio_files = list(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Process each file
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            
            # Create cropped versions
            for i in range(num_crops_per_file):
                # Randomly select segment duration
                segment_duration = np.random.choice(segment_durations)
                
                # Skip if audio is too short
                if len(audio) < segment_duration * sample_rate:
                    print(f"Skipping {audio_path}: too short for {segment_duration}s crop")
                    continue
                
                # Crop the audio
                cropped, start_time, end_time = crop_audio_segment(
                    audio, 
                    sr=sample_rate,
                    segment_duration=segment_duration
                )
                
                # Create output filename
                output_filename = f"{audio_path.stem}_crop_{i+1}_{start_time:.1f}s_to_{end_time:.1f}s.wav"
                output_path = output_dir / output_filename
                
                # Save as WAV
                scipy.io.wavfile.write(
                    output_path,
                    rate=sample_rate,
                    data=(cropped * 32767).astype(np.int16)
                )
                
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
    
    print(f"Completed! Cropped audio files saved to {output_dir}")


# Demo/test code
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    import scipy.io.wavfile
    
    # Check if running as script with arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir + "_cropped"
        
        print(f"Creating cropped versions of audio in {input_dir}")
        print(f"Output will be saved to {output_dir}")
        
        batch_create_cropped_audio(input_dir, output_dir)
    else:
        # Demo with a single file
        print("Running demo with test file...")
        
        # Test with a sample file
        test_file = input("Enter path to a test WAV file: ")
        if os.path.exists(test_file):
            audio, sr = librosa.load(test_file, sr=32000, mono=True)
            print(f"Loaded audio, length: {len(audio)/sr:.2f}s, sample rate: {sr}Hz")
            
            # Create cropped versions
            for i in range(3):
                segment_duration = np.random.choice([3.0, 5.0, 7.0])
                cropped, start_time, end_time = crop_audio_segment(
                    audio, 
                    sr=sr, 
                    segment_duration=segment_duration
                )
                print(f"Created crop {i+1}: {start_time:.1f}s to {end_time:.1f}s ({end_time-start_time:.1f}s)")
                
                # Save to current directory
                output_file = f"test_crop_{i+1}_{start_time:.1f}s_to_{end_time:.1f}s.wav"
                scipy.io.wavfile.write(
                    output_file,
                    rate=sr,
                    data=(cropped * 32767).astype(np.int16)
                )
                print(f"Saved to {output_file}")
        else:
            print(f"File not found: {test_file}")
            print("Usage: python random_cropping.py [input_dir] [output_dir]")