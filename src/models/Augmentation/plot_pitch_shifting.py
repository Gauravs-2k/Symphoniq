#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot waveform and spectrogram comparison between original, pitch-up, and pitch-down audio
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_pitch_comparison_trio(original_path, pitch_up_path, pitch_down_path, output_path=None):
    """
    Generate waveform and spectrogram visualizations comparing original, pitch-up and pitch-down audio
    
    Args:
        original_path: Path to original audio file
        pitch_up_path: Path to pitch-up audio file
        pitch_down_path: Path to pitch-down audio file
        output_path: Path to save the visualization image (optional)
    """
    # Load audio files
    y_original, sr = librosa.load(original_path, sr=None)
    y_pitch_up, _ = librosa.load(pitch_up_path, sr=sr)
    y_pitch_down, _ = librosa.load(pitch_down_path, sr=sr)
    
    # Create time axis in seconds
    time_original = np.linspace(0, len(y_original)/sr, len(y_original))
    time_pitch_up = np.linspace(0, len(y_pitch_up)/sr, len(y_pitch_up))
    time_pitch_down = np.linspace(0, len(y_pitch_down)/sr, len(y_pitch_down))
    
    # Compute spectrograms
    D_original = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_original)), ref=np.max)
    D_pitch_up = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_pitch_up)), ref=np.max)
    D_pitch_down = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_pitch_down)), ref=np.max)
    
    # Create plot with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot original waveform
    axs[0, 0].plot(time_original, y_original, color='blue')
    axs[0, 0].set_title(f'Original Waveform')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    
    # Plot pitch-up waveform
    axs[0, 1].plot(time_pitch_up, y_pitch_up, color='green')
    axs[0, 1].set_title(f'Pitch Up Waveform')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True)
    
    # Plot pitch-down waveform
    axs[0, 2].plot(time_pitch_down, y_pitch_down, color='red')
    axs[0, 2].set_title(f'Pitch Down Waveform')
    axs[0, 2].set_ylabel('Amplitude')
    axs[0, 2].grid(True)
    
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
    
    # Plot pitch-up spectrogram
    img2 = librosa.display.specshow(
        D_pitch_up, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 1]
    )
    axs[1, 1].set_title('Pitch Up Spectrogram')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Frequency (Hz)')
    
    # Plot pitch-down spectrogram
    img3 = librosa.display.specshow(
        D_pitch_down, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        ax=axs[1, 2]
    )
    axs[1, 2].set_title('Pitch Down Spectrogram')
    axs[1, 2].set_xlabel('Time (s)')
    axs[1, 2].set_ylabel('Frequency (Hz)')
    
    # Add colorbars
    fig.colorbar(img1, ax=axs[1, 0], format='%+2.0f dB')
    fig.colorbar(img2, ax=axs[1, 1], format='%+2.0f dB')
    fig.colorbar(img3, ax=axs[1, 2], format='%+2.0f dB')
    
    # Extract pitch shift values from filenames if available
    pitch_up_name = Path(pitch_up_path).stem
    pitch_down_name = Path(pitch_down_path).stem
    
    # Add main title
    plt.suptitle(f'Pitch Shifting Comparison: {Path(original_path).stem}', fontsize=16)
    
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
    pitch_up_file = data_dir / "allemande_first_fragment_pitch_up_1.wav"  # Adjust filename as needed
    pitch_down_file = data_dir / "allemande_first_fragment_pitch_down_1.wav"  # Adjust filename as needed
    
    # Output file path
    output_file = output_dir / "pitch_shift_trio_comparison.png"
    
    # Generate plot
    plot_pitch_comparison_trio(original_file, pitch_up_file, pitch_down_file, output_file)
    
    print("Pitch shift trio visualization complete")