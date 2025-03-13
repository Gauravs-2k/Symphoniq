#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise addition augmentation for audio files
"""

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
from scipy import signal

def add_noise_to_audio(audio: np.ndarray, noise_level: float = 0.005, 
                       noise_type: str = 'white', normalize: bool = True) -> np.ndarray:
    """
    Add noise to audio signal
    
    Args:
        audio: Audio array
        noise_level: Amplitude of noise (0.005 = 0.5% noise)
        noise_type: Type of noise ('white', 'pink', 'brown')
        normalize: Whether to normalize audio after augmentation
        
    Returns:
        Noisy audio as numpy array
    """
    # Make a copy of the audio to avoid modifying the original
    noisy_audio = audio.copy()
    
    # Generate noise based on type
    if noise_type == 'white':
        # White noise - equal energy at all frequencies
        noise = np.random.randn(len(audio)) * noise_level
    
    elif noise_type == 'pink':
        # Pink noise - 1/f spectrum (more bass)
        white_noise = np.random.randn(len(audio))
        # Generate pink noise by filtering white noise
        # This is a simple approximation using scipy.signal instead of librosa.filters
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        noise = signal.filtfilt(b, a, white_noise) * noise_level * 2
    
    elif noise_type == 'brown':
        # Brown noise - 1/f^2 spectrum (even more bass)
        white_noise = np.random.randn(len(audio))
        # Generate brown noise by integrating white noise
        noise = np.cumsum(white_noise) / np.sqrt(len(audio)) * noise_level
        # Remove DC offset
        noise = noise - np.mean(noise)
    
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    # Add noise to audio
    noisy_audio = noisy_audio + noise
    
    # Normalize if requested
    if normalize:
        noisy_audio = librosa.util.normalize(noisy_audio)
    
    return noisy_audio

def create_noisy_prompt(piece_name: str, instrument: str, noise_type: str, level: float) -> str:
    """
    Create a text prompt for examples with added noise
    
    Args:
        piece_name: Name of the piece
        instrument: Instrument name
        noise_type: Type of noise added
        level: Noise level
        
    Returns:
        Text prompt describing the noisy music
    """
    # Create human-readable descriptions
    noise_descriptions = {
        'white': "with some background hiss",
        'pink': "with some room ambience",
        'brown': "with some low frequency rumble"
    }
    
    level_descriptions = {
        0.001: "very subtle",
        0.002: "subtle",
        0.005: "noticeable",
        0.01: "moderate",
        0.02: "significant"
    }
    
    # Get closest level description
    level_keys = np.array(list(level_descriptions.keys()))
    closest_level = level_keys[np.argmin(np.abs(level_keys - level))]
    
    noise_desc = noise_descriptions.get(noise_type, "with some background noise")
    level_desc = level_descriptions.get(closest_level, "noticeable")
    
    prompt = (
        f"Generate a solo {instrument} performance playing {piece_name}. "
        f"The recording has {level_desc} {noise_desc}. "
        f"The {instrument} should play with proper breath control, fluid phrasing, "
        f"clear articulation, and accurate timing despite the noise. This should sound like a "
        f"professional {instrument} recording with natural dynamics and expressive interpretation."
    )
    
    return prompt


def create_noisy_examples(
    audio_files: List[Path], 
    instrument: str,
    noise_types: List[str] = ['white', 'pink', 'brown'],
    noise_levels: List[float] = [0.002, 0.005, 0.01],
    preprocess_audio_file = None,  # Function to preprocess audio file
    desc: str = None
) -> List[Dict[str, Any]]:
    """
    Create noisy versions of audio files
    
    Args:
        audio_files: List of audio file paths
        instrument: Instrument name for prompts
        noise_types: List of noise types to apply
        noise_levels: List of noise levels to apply
        preprocess_audio_file: Function to preprocess audio file
        desc: Description for progress bar
        
    Returns:
        List of examples with noisy audio
    """
    examples = []
    
    # Create description for progress bar if not provided
    if desc is None:
        desc = f"Creating noisy versions"
    
    # Process each noise type and level
    print(f"Adding noise augmentations...")
    for noise_type in noise_types:
        for level in noise_levels:
            augmented_entries = []
            
            for audio_path in tqdm(audio_files, desc=f"{noise_type} noise (level={level:.4f})"):
                try:
                    # Get base name for prompt
                    piece_name = audio_path.stem.replace('_', ' ')
                    
                    # Create prompt
                    prompt = create_noisy_prompt(piece_name, instrument, noise_type, level)
                    
                    # Load original audio
                    audio = preprocess_audio_file(audio_path)
                    
                    # Apply noise
                    noisy_audio = add_noise_to_audio(audio, noise_level=level, noise_type=noise_type)
                    
                    # Add to examples
                    augmented_entries.append({
                        "prompt": prompt,
                        "audio": noisy_audio.tolist(),  # Convert to list for serialization
                        "audio_path": str(audio_path),
                        "augmentation": f"noise_{noise_type}_{level:.4f}"
                    })
                    
                except Exception as e:
                    print(f"Error processing {audio_path} for noise addition: {str(e)}")
            
            examples.extend(augmented_entries)
    
    return examples


# Stand-alone function to batch process a directory
def batch_create_noisy_audio(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path],
    noise_types: List[str] = ['white', 'pink', 'brown'],
    noise_levels: List[float] = [0.002, 0.005, 0.01],
    sample_rate: int = 32000
) -> None:
    """
    Utility function to batch process a directory of audio files and save noisy versions
    
    Args:
        input_dir: Directory containing source audio files
        output_dir: Directory to save noisy audio files
        noise_types: List of noise types to apply
        noise_levels: List of noise levels to apply
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
            
            # Create noisy versions
            for noise_type in noise_types:
                for level in noise_levels:
                    # Apply noise
                    noisy_audio = add_noise_to_audio(audio, noise_level=level, noise_type=noise_type)
                    
                    # Create output filename
                    output_filename = f"{audio_path.stem}_noise_{noise_type}_{level:.4f}.wav"
                    output_path = output_dir / output_filename
                    
                    # Save as WAV
                    scipy.io.wavfile.write(
                        output_path,
                        rate=sample_rate,
                        data=(noisy_audio * 32767).astype(np.int16)
                    )
                    
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
    
    print(f"Completed! Noisy audio files saved to {output_dir}")


# Demo/test code
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    import scipy.io.wavfile
    
    # Check if running as script with arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir + "_noisy"
        
        print(f"Creating noisy versions of audio in {input_dir}")
        print(f"Output will be saved to {output_dir}")
        
        batch_create_noisy_audio(input_dir, output_dir)
    else:
        # Demo with a single file
        print("Running demo with test file...")
        
        # Test with a sample file
        test_file = input("Enter path to a test WAV file: ")
        if os.path.exists(test_file):
            audio, sr = librosa.load(test_file, sr=32000, mono=True)
            print(f"Loaded audio, length: {len(audio)/sr:.2f}s, sample rate: {sr}Hz")
            
            # Create noisy versions
            for noise_type in ['white', 'pink', 'brown']:
                for level in [0.002, 0.01]:
                    noisy = add_noise_to_audio(audio, noise_level=level, noise_type=noise_type)
                    print(f"Created {noise_type} noise version (level {level})")
                    
                    # Save to current directory
                    output_file = f"test_{noise_type}_noise_{level}.wav"
                    scipy.io.wavfile.write(
                        output_file,
                        rate=sr,
                        data=(noisy * 32767).astype(np.int16)
                    )
                    print(f"Saved to {output_file}")
        else:
            print(f"File not found: {test_file}")
            print("Usage: python noise_addition.py [input_dir] [output_dir]")