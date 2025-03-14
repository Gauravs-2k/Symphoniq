#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot waveform and spectrogram comparison between original, time-compressed, and time-expanded audio
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_time_stretching_comparison(original_path, expanded_path, compressed_path, output_path=None):
    """
    Generate waveform and spectrogram visualizations comparing original and time-stretched audio
    
    Args:
        original_path: Path to original audio file
        expanded_path: Path to time-expanded (slower) audio file
        compressed_path: Path to time-compressed (faster) audio file
        output_path: Path to save the visualization image (optional)
    """
    # Load audio files
    y_original, sr = librosa.load(original_path, sr=None)
    y_expanded, _ = librosa.load(expanded_path, sr=sr)
    y_compressed, _ = librosa.load(compressed_path, sr=sr)
    
    # Create time axis in seconds
    time_original = np.linspace(0, len(y_original)/sr, len(y_original))
    time_expanded = np.linspace(0, len(y_expanded)/sr, len(y_expanded))
    time_compressed = np.linspace(0, len(y_compressed)/sr, len(y_compressed))
    
    # Calculate durations
    original_duration = len(y_original) / sr
    expanded_duration = len(y_expanded) / sr
    compressed_duration = len(y_compressed) / sr
    
    # Calculate stretch ratios
    expand_ratio = expanded_duration / original_duration
    compress_ratio = compressed_duration / original_duration
    
    # Compute spectrograms
    D_original = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_original)), ref=np.max)
    D_expanded = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_expanded)), ref=np.max)
    D_compressed = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_compressed)), ref=np.max)
    
    # Create plot with 2 rows and 3 columns - increase figure size for better spacing
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot original waveform
    axs[0, 0].plot(time_original, y_original, color='blue')
    axs[0, 0].set_title('Original Waveform', fontsize=12)
    axs[0, 0].set_ylabel('Amplitude', fontsize=11)
    axs[0, 0].text(
        0.5, -0.15, 
        f'Duration: {original_duration:.2f}s', 
        transform=axs[0, 0].transAxes,
        ha='center', fontsize=10
    )
    axs[0, 0].grid(True)
    
    # Plot expanded waveform
    axs[0, 1].plot(time_expanded, y_expanded, color='purple')
    axs[0, 1].set_title('Expanded Waveform', fontsize=12)
    axs[0, 1].text(
        0.5, -0.15, 
        f'Duration: {expanded_duration:.2f}s (Ratio: {expand_ratio:.2f}x)', 
        transform=axs[0, 1].transAxes,
        ha='center', fontsize=10
    )
    axs[0, 1].grid(True)
    
    # Plot compressed waveform
    axs[0, 2].plot(time_compressed, y_compressed, color='green')
    axs[0, 2].set_title('Compressed Waveform', fontsize=12)
    axs[0, 2].text(
        0.5, -0.15, 
        f'Duration: {compressed_duration:.2f}s (Ratio: {compress_ratio:.2f}x)', 
        transform=axs[0, 2].transAxes,
        ha='center', fontsize=10
    )
    axs[0, 2].grid(True)
    
    # Add a single y-label for the top row
    fig.text(0.04, 0.75, 'Amplitude', va='center', rotation='vertical', fontsize=12)
    
    # Plot original spectrogram
    img1 = librosa.display.specshow(
        D_original, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 0]
    )
    axs[1, 0].set_title('Original Spectrogram', fontsize=12)
    axs[1, 0].set_xlabel('Time (s)', fontsize=11)
    
    # Plot expanded spectrogram
    img2 = librosa.display.specshow(
        D_expanded, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 1]
    )
    axs[1, 1].set_title('Expanded Spectrogram', fontsize=12)
    axs[1, 1].set_xlabel('Time (s)', fontsize=11)
    
    # Plot compressed spectrogram
    img3 = librosa.display.specshow(
        D_compressed, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 2]
    )
    axs[1, 2].set_title('Compressed Spectrogram', fontsize=12)
    axs[1, 2].set_xlabel('Time (s)', fontsize=11)
    
    # Add a single y-label for the bottom row
    fig.text(0.04, 0.3, 'Frequency (Hz)', va='center', rotation='vertical', fontsize=12)
    
    # Add colorbars
    cbar1 = fig.colorbar(img1, ax=axs[1, 0], format='%+2.0f dB', shrink=0.8)
    cbar2 = fig.colorbar(img2, ax=axs[1, 1], format='%+2.0f dB', shrink=0.8)
    cbar3 = fig.colorbar(img3, ax=axs[1, 2], format='%+2.0f dB', shrink=0.8)
    
    # Add main title
    plt.suptitle('Time Stretching Comparison', fontsize=16, y=0.98)
    
    # Add file information as subtitle
    plt.figtext(
        0.5, 0.92, 
        f"Original: {Path(original_path).stem}",
        ha='center', fontsize=10, style='italic'
    )
    
    # Add more space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.9, bottom=0.1, left=0.1, right=0.95)
    
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
    expanded_file = data_dir / "allemande_fifth_fragment_tempo_slower_1.11.wav"  # Slower version
    compressed_file = data_dir / "allemande_fifth_fragment_tempo_faster_1.1.wav"  # Faster version
    
    # Output file path
    output_file = output_dir / "time_stretching_comparison.png"
    
    # Generate plot
    plot_time_stretching_comparison(original_file, expanded_file, compressed_file, output_file)
    
    print("Time stretching visualization complete")