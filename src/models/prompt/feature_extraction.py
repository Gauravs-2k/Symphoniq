import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional


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
        
        # Plot spectrogram
        plt.subplot(3, 1, 1)
        librosa.display.specshow(
            features['spectrogram'], sr=features['sample_rate'], 
            hop_length=512, x_axis='time', y_axis='hz'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # Plot mel spectrogram
        plt.subplot(3, 1, 2)
        librosa.display.specshow(
            features['mel_spectrogram'], sr=features['sample_rate'], 
            hop_length=512, x_axis='time', y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        # Plot MFCCs
        plt.subplot(3, 1, 3)
        librosa.display.specshow(
            features['mfcc'], sr=features['sample_rate'], 
            hop_length=512, x_axis='time'
        )
        plt.colorbar()
        plt.title('MFCC')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
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
    return extractor.extract_all_features(file_path)
