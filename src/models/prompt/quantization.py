import numpy as np
from typing import Dict, Any, List
import librosa
from sklearn.cluster import KMeans


class FeatureQuantizer:
    """Quantizes audio features into simplified token sequences"""
    
    def __init__(self, n_clusters=16, n_levels=32):
        self.n_clusters = n_clusters  # For MFCC clustering
        self.n_levels = n_levels      # For mel spectrogram quantization
    
    def quantize_mfcc(self, mfcc: np.ndarray) -> np.ndarray:
        """Convert MFCCs to cluster IDs using k-means"""
        # Transpose to get time frames as samples
        mfcc_frames = mfcc.T
        
        # Train k-means on the MFCC frames
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(mfcc_frames)
        
        return cluster_ids
    
    def quantize_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """Quantize mel spectrogram values to discrete levels"""
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
        """Extract and quantize the main pitch contour"""
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


def quantize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to quantize a feature dictionary"""
    quantizer = FeatureQuantizer()
    return quantizer.quantize_all(features)