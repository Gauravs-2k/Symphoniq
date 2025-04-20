import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
import os


class VocalFeatureExtractor:
    """
    A class for extracting audio features from vocal WAV files.
    Supports extraction of spectrograms, mel spectrograms, and MFCCs.
    """
    
    def __init__(self, sample_rate: Optional[int] = None):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate. If None, uses the native sample rate.
        """
        self.sample_rate = sample_rate
        self.audio_data = None
        self.sr = None
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file using librosa.
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            Tuple of audio data and sample rate
        """
        self.audio_data, self.sr = librosa.load(file_path, sr=self.sample_rate)
        return self.audio_data, self.sr
    
    def extract_spectrogram(self, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Extract spectrogram from loaded audio.
        
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            
        Returns:
            Spectrogram as a numpy array
        """
        if self.audio_data is None:
            raise ValueError("No audio data loaded. Call load_audio first.")
            
        # Compute the spectrogram magnitude
        spectrogram = np.abs(librosa.stft(self.audio_data, n_fft=n_fft, hop_length=hop_length))
        
        # Convert to dB scale
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        return spectrogram_db
    
    def extract_mel_spectrogram(self, n_fft: int = 2048, hop_length: int = 512, 
                                n_mels: int = 128) -> np.ndarray:
        """
        Extract mel spectrogram from loaded audio.
        
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel bands
            
        Returns:
            Mel spectrogram as a numpy array
        """
        if self.audio_data is None:
            raise ValueError("No audio data loaded. Call load_audio first.")
            
        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=self.audio_data, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        
        # Convert to dB scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db
    
    def extract_mfcc(self, n_mfcc: int = 13, n_fft: int = 2048, 
                     hop_length: int = 512) -> np.ndarray:
        """
        Extract MFCCs from loaded audio.
        
        Args:
            n_mfcc: Number of MFCCs to return
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            
        Returns:
            MFCCs as a numpy array
        """
        if self.audio_data is None:
            raise ValueError("No audio data loaded. Call load_audio first.")
            
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=self.audio_data, sr=self.sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        return mfccs
    
    def extract_all_features(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all audio features from a file.
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load the audio file
        self.load_audio(file_path)
        
        # Extract features
        features = {
            'audio_data': self.audio_data,
            'sample_rate': self.sr,
            'spectrogram': self.extract_spectrogram(),
            'mel_spectrogram': self.extract_mel_spectrogram(),
            'mfcc': self.extract_mfcc()
        }
        
        return features
    
    def plot_features(self, features: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot extracted features for visualization.
        
        Args:
            features: Dictionary of features obtained from extract_all_features
            save_path: If provided, save the figure to this path
        """
        plt.figure(figsize=(15, 10))
        plt.suptitle('Extracted Vocal Audio Features', fontsize=16)
        
        # Plot spectrogram
        plt.subplot(3, 1, 1)
        librosa.display.specshow(
            features['spectrogram'], sr=features['sample_rate'], 
            hop_length=512, x_axis='time', y_axis='hz'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Frequency Spectrogram (dB)')
        
        # Plot mel spectrogram
        plt.subplot(3, 1, 2)
        librosa.display.specshow(
            features['mel_spectrogram'], sr=features['sample_rate'], 
            hop_length=512, x_axis='time', y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-scale Spectrogram (dB)')
        
        # Plot MFCCs
        plt.subplot(3, 1, 3)
        librosa.display.specshow(
            features['mfcc'], sr=features['sample_rate'], 
            hop_length=512, x_axis='time'
        )
        plt.colorbar()
        plt.title('Mel-Frequency Cepstral Coefficients (MFCC)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_feature_distributions(self, features: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot distributions of extracted features as bar charts.
        
        Args:
            features: Dictionary of features obtained from extract_all_features
            save_path: If provided, save the figure to this path
        """
        plt.figure(figsize=(15, 12))
        plt.suptitle('Vocal Feature Value Distributions', fontsize=16)
        
        # Plot spectrogram distribution
        plt.subplot(3, 1, 1)
        spec_data = features['spectrogram'].flatten()
        plt.hist(spec_data, bins=50, alpha=0.7, color='blue')
        plt.title('Frequency Spectrogram Value Distribution')
        plt.xlabel('Decibel Value (dB)')
        plt.ylabel('Frequency Count')
        plt.grid(True, alpha=0.3)
        
        # Plot mel spectrogram distribution
        plt.subplot(3, 1, 2)
        mel_data = features['mel_spectrogram'].flatten()
        plt.hist(mel_data, bins=50, alpha=0.7, color='green')
        plt.title('Mel-scale Spectrogram Value Distribution')
        plt.xlabel('Decibel Value (dB)')
        plt.ylabel('Frequency Count')
        plt.grid(True, alpha=0.3)
        
        # Plot MFCC distribution
        plt.subplot(3, 1, 3)
        mfcc_data = features['mfcc'].flatten()
        plt.hist(mfcc_data, bins=50, alpha=0.7, color='red')
        plt.title('MFCC Coefficient Value Distribution')
        plt.xlabel('MFCC Coefficient Value')
        plt.ylabel('Frequency Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Vocal feature distribution chart saved to: {save_path}")
        
        plt.show()


def extract_features_from_file(file_path: str) -> Dict[str, Any]:
    """
    Utility function to extract all features from a WAV file.
    
    Args:
        file_path: Path to the WAV file
        
    Returns:
        Dictionary containing all extracted features
    """
    extractor = VocalFeatureExtractor()
    features = extractor.extract_all_features(file_path)
    
    # Save feature distribution chart
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(os.path.dirname(file_path), "analysis")
    os.makedirs(save_dir, exist_ok=True)
    distribution_path = os.path.join(save_dir, f"{base_name}_feature_distribution.png")
    
    extractor.plot_feature_distributions(features, distribution_path)
    
    return features
