#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot waveform and spectrogram comparison between original and randomly cropped audio
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as patches

def plot_random_cropping_comparison(original_path, cropped_path, output_path=None, crop_start=None, crop_end=None):
    """
    Generate waveform and spectrogram visualizations comparing original and cropped audio
    
    Args:
        original_path: Path to original audio file
        cropped_path: Path to cropped audio file
        output_path: Path to save the visualization image (optional)
        crop_start: Starting point of crop in seconds (if known)
        crop_end: Ending point of crop in seconds (if known)
    """
    # Load audio files
    y_original, sr = librosa.load(original_path, sr=None)
    y_cropped, _ = librosa.load(cropped_path, sr=sr)
    
    # Create time axis in seconds
    time_original = np.linspace(0, len(y_original)/sr, len(y_original))
    time_cropped = np.linspace(0, len(y_cropped)/sr, len(y_cropped))
    
    # Calculate duration
    original_duration = len(y_original) / sr
    cropped_duration = len(y_cropped) / sr
    
    # Compute spectrograms
    D_original = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_original)), ref=np.max)
    D_cropped = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_cropped)), ref=np.max)
    
    # Create plot with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot original waveform
    axs[0, 0].plot(time_original, y_original, color='blue')
    axs[0, 0].set_title(f'Original Waveform ({original_duration:.2f}s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    
    # Add crop region indicator if crop boundaries are provided
    if crop_start is not None and crop_end is not None:
        crop_rect = patches.Rectangle(
            (crop_start, -1), crop_end - crop_start, 2, 
            linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.3
        )
        axs[0, 0].add_patch(crop_rect)
        axs[0, 0].text(
            crop_start + (crop_end - crop_start) / 2, 
            0.8, 
            'Cropped Region', 
            color='orange',
            fontweight='bold',
            ha='center'
        )
    
    # Plot cropped waveform
    axs[0, 1].plot(time_cropped, y_cropped, color='orange')
    axs[0, 1].set_title(f'Cropped Waveform ({cropped_duration:.2f}s)')
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
    axs[1, 0].set_title('Original Spectrogram')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    
    # Plot cropped spectrogram
    img2 = librosa.display.specshow(
        D_cropped, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 1]
    )
    axs[1, 1].set_title('Cropped Spectrogram')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Frequency (Hz)')
    
    # Add colorbars
    fig.colorbar(img1, ax=axs[1, 0], format='%+2.0f dB')
    fig.colorbar(img2, ax=axs[1, 1], format='%+2.0f dB')
    
    # Add main title
    plt.suptitle(
        f'Random Cropping Comparison: {Path(original_path).stem}\n'
        f'Original: {original_duration:.2f}s → Cropped: {cropped_duration:.2f}s', 
        fontsize=16
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
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
    cropped_file = data_dir / "allemande_first_fragment_crop_1_15.3s_to_18.3s.wav"  # Adjust filename as needed
    
    # Output file path
    output_file = output_dir / "random_cropping_comparison.png"
    
    # Optional: If you know the cropping points (in seconds)
    crop_start = None  # Example: 2.5
    crop_end = None    # Example: 7.2
    
    # Generate plot
    plot_random_cropping_comparison(
        original_file, 
        cropped_file, 
        output_file, 
        crop_start=crop_start, 
        crop_end=crop_end
    )
    
    print("Random cropping visualization complete")