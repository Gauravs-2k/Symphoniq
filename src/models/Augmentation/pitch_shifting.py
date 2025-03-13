#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch shifting augmentation for audio files
"""

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from typing import List, Dict, Any, Optional, Union

def pitch_shift_audio(audio: np.ndarray, sr: int = 32000, n_steps: float = 1.0, 
                      normalize: bool = True) -> np.ndarray:
    """
    Apply pitch shifting to audio
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_steps: Number of semitones to shift pitch (positive or negative)
        normalize: Whether to normalize audio after augmentation
        
    Returns:
        Pitch-shifted audio as numpy array
    """
    # Apply pitch shifting
    shifted_audio = librosa.effects.pitch_shift(
        audio, 
        sr=sr, 
        n_steps=n_steps
    )
    
    # Normalize if requested
    if normalize:
        shifted_audio = librosa.util.normalize(shifted_audio)
        
    return shifted_audio


def create_pitch_shifted_prompt(piece_name: str, instrument: str, semitones: float) -> str:
    """
    Create a text prompt for pitch-shifted examples
    
    Args:
        piece_name: Name of the piece
        instrument: Instrument name
        semitones: Number of semitones shifted
        
    Returns:
        Text prompt describing the pitch-shifted music
    """
    direction = "higher" if semitones > 0 else "lower"
    shift_desc = f"{abs(semitones)} semitone{'s' if abs(semitones) > 1 else ''} {direction}"
    
    prompt = (
        f"Generate a solo {instrument} performance playing {piece_name}. "
        f"Play this {shift_desc} than standard pitch. "
        f"The {instrument} should play with proper breath control, fluid phrasing, "
        f"clear articulation, and accurate timing. This should sound like a professional "
        f"{instrument} recording with natural dynamics and expressive interpretation."
    )
    
    return prompt


def create_pitch_shifted_examples(
    audio_files: List[Path], 
    instrument: str,
    semitones: List[float],
    preprocess_audio_file,  # Function to preprocess audio file
    desc: str = None
) -> List[Dict[str, Any]]:
    """
    Create pitch-shifted versions of audio files
    
    Args:
        audio_files: List of audio file paths
        instrument: Instrument name for prompts
        semitones: List of semitone shifts to apply
        preprocess_audio_file: Function to preprocess audio file
        desc: Description for progress bar
        
    Returns:
        List of examples with pitch-shifted audio
    """
    examples = []
    
    # Create description for progress bar if not provided
    if desc is None:
        steps_str = ", ".join([str(s) for s in semitones])
        desc = f"Creating pitch-shifted versions ({steps_str} semitones)"
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc=desc):
        try:
            # Get base name for prompt
            piece_name = audio_path.stem.replace('_', ' ')
            
            # Load original audio
            audio = preprocess_audio_file(audio_path)
            
            # Create shifted versions for each requested semitone value
            for steps in semitones:
                # Skip 0 (no shift)
                if steps == 0:
                    continue
                
                # Create prompt
                prompt = create_pitch_shifted_prompt(piece_name, instrument, steps)
                
                # Apply pitch shifting
                shifted_audio = pitch_shift_audio(audio, n_steps=steps)
                
                # Add to examples
                examples.append({
                    "prompt": prompt,
                    "audio": shifted_audio.tolist(),  # Convert to list for serialization
                    "audio_path": str(audio_path),
                    "augmentation": f"pitch_shift_{steps}"
                })
                
        except Exception as e:
            print(f"Error processing {audio_path} for pitch shifting: {str(e)}")
    
    return examples


# Stand-alone function to batch process a directory
def batch_create_pitch_shifted_audio(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path],
    semitones: List[float] = [-2, -1, 1, 2],
    sample_rate: int = 32000
) -> None:
    """
    Utility function to batch process a directory of audio files and save pitch-shifted versions
    
    Args:
        input_dir: Directory containing source audio files
        output_dir: Directory to save pitch-shifted audio files
        semitones: List of semitone shifts to apply
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
            
            # Create shifted versions
            for steps in semitones:
                # Skip 0 (no shift)
                if steps == 0:
                    continue
                
                # Apply pitch shifting
                shifted_audio = pitch_shift_audio(audio, sr=sample_rate, n_steps=steps)
                
                # Create output filename
                shift_str = f"up_{steps}" if steps > 0 else f"down_{abs(steps)}"
                output_filename = f"{audio_path.stem}_pitch_{shift_str}.wav"
                output_path = output_dir / output_filename
                
                # Save as WAV
                scipy.io.wavfile.write(
                    output_path,
                    rate=sample_rate,
                    data=(shifted_audio * 32767).astype(np.int16)
                )
                
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
    
    print(f"Completed! Pitch-shifted audio files saved to {output_dir}")


# Demo/test code
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    import scipy.io.wavfile
    
    # Check if running as script with arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir + "_shifted"
        semitones = [-2, -1, 1, 2]  # Default pitch shifts
        
        print(f"Creating pitch-shifted versions of audio in {input_dir}")
        print(f"Output will be saved to {output_dir}")
        
        batch_create_pitch_shifted_audio(input_dir, output_dir, semitones)
    else:
        # Demo with a single file
        print("Running demo with test file...")
        
        # Test with a sample file
        test_file = input("Enter path to a test WAV file: ")
        if os.path.exists(test_file):
            audio, sr = librosa.load(test_file, sr=32000, mono=True)
            print(f"Loaded audio, length: {len(audio)/sr:.2f}s, sample rate: {sr}Hz")
            
            # Create shifted versions
            for steps in [-2, -1, 1, 2]:
                shifted = pitch_shift_audio(audio, sr, steps)
                print(f"Created pitch-shifted version ({steps} semitones)")
                
                # Save to current directory
                output_file = f"test_shifted_{steps}_semitones.wav"
                scipy.io.wavfile.write(
                    output_file,
                    rate=sr,
                    data=(shifted * 32767).astype(np.int16)
                )
                print(f"Saved to {output_file}")
        else:
            print(f"File not found: {test_file}")
            print("Usage: python pitch_shifting.py [input_dir] [output_dir]")