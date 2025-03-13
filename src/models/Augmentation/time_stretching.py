#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time stretching augmentation for audio files
"""

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union

def time_stretch_audio(audio: np.ndarray, rate: float = 1.0, normalize: bool = True) -> np.ndarray:
    """
    Apply time stretching to audio
    
    Args:
        audio: Audio array
        rate: Time stretch factor (>1 speeds up, <1 slows down)
        normalize: Whether to normalize audio after augmentation
        
    Returns:
        Time-stretched audio as numpy array
    """
    # Apply time stretching
    stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
    
    # Normalize if requested
    if normalize:
        stretched_audio = librosa.util.normalize(stretched_audio)
        
    return stretched_audio


def create_time_stretched_prompt(piece_name: str, instrument: str, factor: float) -> str:
    """
    Create a text prompt for time-stretched examples
    
    Args:
        piece_name: Name of the piece
        instrument: Instrument name
        factor: Time stretch factor
        
    Returns:
        Text prompt describing the time-stretched music
    """
    tempo_desc = "faster" if factor > 1 else "slower"
    percent_change = abs(100 - (factor * 100))
    
    prompt = (
        f"Generate a solo {instrument} performance playing {piece_name}. "
        f"Play this at a {tempo_desc} tempo (about {percent_change:.0f}% {tempo_desc}). "
        f"The {instrument} should play with proper breath control, fluid phrasing, "
        f"clear articulation, and accurate timing. This should sound like a professional "
        f"{instrument} recording with natural dynamics and expressive interpretation."
    )
    
    return prompt


def create_time_stretched_examples(
    audio_files: List[Path], 
    instrument: str,
    factors: List[float],
    preprocess_audio_file,  # Function to preprocess audio file
    desc: str = None
) -> List[Dict[str, Any]]:
    """
    Create time-stretched versions of audio files
    
    Args:
        audio_files: List of audio file paths
        instrument: Instrument name for prompts
        factors: List of time stretch factors to apply (>1 speeds up, <1 slows down)
        preprocess_audio_file: Function to preprocess audio file
        desc: Description for progress bar
        
    Returns:
        List of examples with time-stretched audio
    """
    examples = []
    
    # Create description for progress bar if not provided
    if desc is None:
        factors_str = ", ".join([str(s) for s in factors])
        desc = f"Creating time-stretched versions ({factors_str}x)"
    
    # Process each audio file
    for factor in factors:
        augmented_entries = []
        tempo_change = "faster" if factor > 1 else "slower"
        
        for audio_path in tqdm(audio_files, desc=f"Time stretch: {factor}x"):
            try:
                # Get base name for prompt
                piece_name = audio_path.stem.replace('_', ' ')
                
                # Create prompt
                prompt = create_time_stretched_prompt(piece_name, instrument, factor)
                
                # Load original audio
                audio = preprocess_audio_file(audio_path)
                
                # Apply time stretching
                stretched_audio = time_stretch_audio(audio, rate=factor)
                
                # Add to examples
                augmented_entries.append({
                    "prompt": prompt,
                    "audio": stretched_audio.tolist(),  # Convert to list for serialization
                    "audio_path": str(audio_path),
                    "augmentation": f"time_stretch_{factor}"
                })
                
            except Exception as e:
                print(f"Error processing {audio_path} for time stretching: {str(e)}")
        
        examples.extend(augmented_entries)
    
    return examples


# Stand-alone function to batch process a directory
def batch_create_time_stretched_audio(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path],
    factors: List[float] = [0.8, 0.9, 1.1, 1.2],
    sample_rate: int = 32000
) -> None:
    """
    Utility function to batch process a directory of audio files and save time-stretched versions
    
    Args:
        input_dir: Directory containing source audio files
        output_dir: Directory to save time-stretched audio files
        factors: List of time stretch factors to apply
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
            
            # Create stretched versions
            for factor in factors:
                # Skip 1.0 (no stretch)
                if factor == 1.0:
                    continue
                
                # Apply time stretching
                stretched_audio = time_stretch_audio(audio, rate=factor)
                
                # Create output filename
                speed_str = f"faster_{factor}" if factor > 1 else f"slower_{1/factor}"
                output_filename = f"{audio_path.stem}_tempo_{speed_str}.wav"
                output_path = output_dir / output_filename
                
                # Save as WAV
                scipy.io.wavfile.write(
                    output_path,
                    rate=sample_rate,
                    data=(stretched_audio * 32767).astype(np.int16)
                )
                
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
    
    print(f"Completed! Time-stretched audio files saved to {output_dir}")


# Demo/test code
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    import scipy.io.wavfile
    
    # Check if running as script with arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir + "_stretched"
        factors = [0.8, 0.9, 1.1, 1.2]  # Default time stretch factors
        
        print(f"Creating time-stretched versions of audio in {input_dir}")
        print(f"Output will be saved to {output_dir}")
        
        batch_create_time_stretched_audio(input_dir, output_dir, factors)
    else:
        # Demo with a single file
        print("Running demo with test file...")
        
        # Test with a sample file
        test_file = input("Enter path to a test WAV file: ")
        if os.path.exists(test_file):
            audio, sr = librosa.load(test_file, sr=32000, mono=True)
            print(f"Loaded audio, length: {len(audio)/sr:.2f}s, sample rate: {sr}Hz")
            
            # Create stretched versions
            for factor in [0.8, 0.9, 1.1, 1.2]:
                stretched = time_stretch_audio(audio, rate=factor)
                tempo_change = "faster" if factor > 1 else "slower"
                print(f"Created {tempo_change} version (factor {factor})")
                
                # Save to current directory
                output_file = f"test_{tempo_change}_{factor}.wav"
                scipy.io.wavfile.write(
                    output_file,
                    rate=sr,
                    data=(stretched * 32767).astype(np.int16)
                )
                print(f"Saved to {output_file}")
        else:
            print(f"File not found: {test_file}")
            print("Usage: python time_stretching.py [input_dir] [output_dir]")