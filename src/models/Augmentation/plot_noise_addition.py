#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot waveform and spectrogram comparison between original and noise-augmented audio
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_audio_comparison(original_path, augmented_path, output_path=None):
    """
    Generate waveform and spectrogram visualizations comparing original and augmented audio
    
    Args:
        original_path: Path to original audio file
        augmented_path: Path to augmented audio file
        output_path: Path to save the visualization image (optional)
    """
    # Load audio files
    y_original, sr = librosa.load(original_path, sr=None)
    y_augmented, _ = librosa.load(augmented_path, sr=sr)
    
    # Create time axis in seconds
    time_original = np.linspace(0, len(y_original)/sr, len(y_original))
    time_augmented = np.linspace(0, len(y_augmented)/sr, len(y_augmented))
    
    # Compute spectrograms
    D_original = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_original)), ref=np.max)
    D_augmented = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_augmented)), ref=np.max)
    
    # Create plot with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot original waveform
    axs[0, 0].plot(time_original, y_original, color='blue')
    axs[0, 0].set_title(f'Original Audio Waveform: {Path(original_path).stem}')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    
    # Plot augmented waveform
    axs[0, 1].plot(time_augmented, y_augmented, color='red')
    axs[0, 1].set_title(f'Augmented Audio Waveform: {Path(augmented_path).stem}')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True)
    
    # Plot original spectrogram
    img1 = librosa.display.specshow(
        D_original, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 0]
    )
    axs[1, 0].set_title('Original Audio Spectrogram')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    
    # Plot augmented spectrogram
    img2 = librosa.display.specshow(
        D_augmented, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 1]
    )
    axs[1, 1].set_title('Augmented Audio Spectrogram')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Frequency (Hz)')
    
    # Add colorbar
    fig.colorbar(img1, ax=axs[1, 0], format='%+2.0f dB')
    fig.colorbar(img2, ax=axs[1, 1], format='%+2.0f dB')
    
    # Add main title
    plt.suptitle('Audio Comparison: Waveform and Spectrogram Analysis', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    # Define paths
    data_dir = Path("/Users/gauravs/Documents/Symphoniq/src/data/augmented/audio")
    output_dir = Path("/Users/gauravs/Documents/Symphoniq/src/models/Augmentation/plots")
    output_dir.mkdir(exist_ok=True)
    
    # Sample files to compare
    original_file = data_dir / "allemande_fifth_fragment_original.wav"
    augmented_file = data_dir / "allemande_first_fragment_noise_white_0.0020.wav"
    
    # Output file path
    output_file = output_dir / "audio_comparison_with_spectrograms.png"
    
    # Generate plot
    plot_audio_comparison(original_file, augmented_file, output_file)
    
    print("Plot generation complete")