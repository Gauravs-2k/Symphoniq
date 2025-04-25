import numpy as np
from typing import Dict, Any, List, Optional
import librosa
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


class FeatureQuantizer:
    """Quantizes continuous audio features into discrete tokens for sequence modeling"""
    
    def __init__(self, n_clusters=16, n_levels=32):
        self.n_clusters = n_clusters  # For MFCC clustering
        self.n_levels = n_levels      # For mel spectrogram quantization
    
    def quantize_mfcc(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Groups similar vocal timbres together using k-means clustering.
        Each MFCC frame gets assigned to one of n_clusters timbral categories.
        This effectively tokenizes the continuously varying voice character.
        """
        # Transpose to get time frames as samples
        mfcc_frames = mfcc.T
        
        # Train k-means on the MFCC frames
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(mfcc_frames)
        
        return cluster_ids
    
    def quantize_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Converts continuous spectral energy measurements into discrete intensity levels.
        Steps:
        1. Normalize the energy to 0-1 range for consistent scaling
        2. Divide into n_levels discrete steps
        3. Summarize each time frame by its most common energy level
        This creates a simplified representation of the spectral shape over time.
        """
        # Normalize to [0,1] range
        mel_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-9)
        
        # Quantize to n_levels
        quantized = np.floor(mel_norm * self.n_levels).astype(np.int32)
        
        # Optional: create a more compact representation by summarizing each time frame
        # Take the most common value in each time frame
        frame_tokens = np.array([
            np.bincount(column, minlength=self.n_levels+1).argmax() 
            for column in quantized.T
        ])
        
        return frame_tokens
    
    def extract_pitch_contour(self, audio: np.ndarray, sr: int) -> List[int]:
        """
        Extracts the main melody line as a sequence of discrete pitch values:
        1. Uses PYIN for robust pitch detection in vocals
        2. Converts frequencies to MIDI note numbers for even pitch spacing
        3. Marks unvoiced segments as -1 (when no clear pitch is detected)
        4. Downsamples to reduce redundancy in the pitch sequence
        """
        # Extract pitch using PYIN algorithm
        f0, voiced_flag, _ = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Remove unvoiced sections (replace with -1)
        f0[~voiced_flag] = -1
        
        # Convert Hz to MIDI note numbers and quantize to semitones
        midi_notes = np.zeros_like(f0, dtype=np.int32)
        voiced_idx = voiced_flag.nonzero()[0]
        if len(voiced_idx) > 0:
            midi_notes[voiced_idx] = np.round(librosa.hz_to_midi(f0[voiced_idx])).astype(np.int32)
        
        # Downsample for a more compact representation (every 10th frame)
        downsampled = midi_notes[::10].tolist()
        
        return downsampled
    
    def extract_rhythm_tokens(self, audio: np.ndarray, sr: int) -> List[int]:
        """Extract and quantize rhythmic patterns"""
        # Get onset strength envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Find beat positions
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Quantize beat strengths on a scale of 0-4
        beat_strengths = onset_env[beats]
        normalized = (beat_strengths - beat_strengths.min()) / (beat_strengths.max() - beat_strengths.min() + 1e-9)
        quantized_strengths = np.floor(normalized * 4).astype(np.int32).tolist()
        
        return quantized_strengths
    
    def quantize_all(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize all features in a feature dictionary"""
        audio = features['audio_data']
        sr = features['sample_rate']
        
        quantized = {
            'mfcc_tokens': self.quantize_mfcc(features['mfcc']),
            'mel_tokens': self.quantize_mel_spectrogram(features['mel_spectrogram']),
            'pitch_tokens': self.extract_pitch_contour(audio, sr),
            'rhythm_tokens': self.extract_rhythm_tokens(audio, sr)
        }
        
        return quantized
    
    def plot_quantized_histograms(self, quantized: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot histograms of the quantized features.
        
        Args:
            quantized: Dictionary of quantized features from quantize_all
            save_path: If provided, save the figure to this path
        """
        plt.figure(figsize=(15, 12))
        plt.suptitle('Quantized Vocal Feature Distributions', fontsize=16)
        
        # Plot MFCC clusters histogram
        plt.subplot(2, 2, 1)
        mfcc_tokens = quantized['mfcc_tokens']
        plt.hist(mfcc_tokens, bins=self.n_clusters, range=(0, self.n_clusters-1), 
                 alpha=0.7, color='blue', edgecolor='black')
        plt.title('MFCC Cluster Distribution')
        plt.xlabel('Cluster ID')
        plt.ylabel('Frequency Count')
        plt.xticks(range(0, self.n_clusters, 2))
        plt.grid(True, alpha=0.3)
        
        # Plot Mel Spectrogram tokens histogram
        plt.subplot(2, 2, 2)
        mel_tokens = quantized['mel_tokens']
        plt.hist(mel_tokens, bins=self.n_levels, range=(0, self.n_levels-1), 
                 alpha=0.7, color='green', edgecolor='black')
        plt.title('Mel Spectrogram Token Distribution')
        plt.xlabel('Quantized Level')
        plt.ylabel('Frequency Count')
        plt.grid(True, alpha=0.3)
        
        # Plot pitch tokens histogram - filter out -1 values (unvoiced)
        plt.subplot(2, 2, 3)
        pitch_tokens = np.array(quantized['pitch_tokens'])
        voiced_pitches = pitch_tokens[pitch_tokens >= 0]  # Filter out unvoiced (-1)
        if len(voiced_pitches) > 0:
            plt.hist(voiced_pitches, bins=min(30, len(np.unique(voiced_pitches))), 
                     alpha=0.7, color='red', edgecolor='black')
            plt.title('Pitch Contour Distribution (MIDI notes)')
            plt.xlabel('MIDI Note Number')
            plt.ylabel('Frequency Count')
        else:
            plt.text(0.5, 0.5, 'No voiced pitches detected', 
                     horizontalalignment='center', verticalalignment='center')
            plt.title('Pitch Contour (No Data)')
        plt.grid(True, alpha=0.3)
        
        # Plot rhythm tokens histogram
        plt.subplot(2, 2, 4)
        rhythm_tokens = quantized['rhythm_tokens']
        plt.hist(rhythm_tokens, bins=5, range=(0, 4), 
                 alpha=0.7, color='purple', edgecolor='black')
        plt.title('Beat Strength Distribution')
        plt.xlabel('Quantized Beat Strength')
        plt.ylabel('Frequency Count')
        plt.xticks(range(5))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Quantized feature histogram saved to: {save_path}")
        
        plt.show()


def quantize_features(features: Dict[str, Any], plot: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to quantize a feature dictionary and optionally plot histograms.
    
    Args:
        features: Dictionary of extracted audio features
        plot: Whether to plot the quantized feature histograms
        save_path: Path to save the histogram if plotting
    
    Returns:
        Dictionary of quantized features
    """
    quantizer = FeatureQuantizer()
    quantized = quantizer.quantize_all(features)
    
    # Always create the plot, but only show it if plot=True
    # This ensures the histogram is always saved
    print(f"Generating quantization histograms...")
    
    # Ensure save_path has a valid directory
    if save_path:
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(save_path)
            if not directory:  # If dirname returns empty string, use current directory
                directory = "."
            
            os.makedirs(directory, exist_ok=True)
            print(f"Ensuring directory exists: {directory}")
            
            # Generate the plot and save it
            quantizer.plot_quantized_histograms(quantized, save_path)
            print(f"Successfully saved quantization histogram to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
            # If there's an error saving, still try to generate the plot without saving
            if plot:
                quantizer.plot_quantized_histograms(quantized, None)
    elif plot:
        # If no save path but plot is true, just display it
        quantizer.plot_quantized_histograms(quantized, None)
    
    return quantized