from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
import random  # Import random for variety

class PromptGenerator:
    """Converts quantized audio features into text prompts for MusicGen"""
    
    def __init__(self):
        # Maps pitch ranges to descriptive terms
        self.pitch_range_map = {
            (0, 48): "very low-pitched",
            (48, 60): "low-pitched",
            (60, 72): "mid-range",
            (72, 84): "high-pitched",
            (84, 128): "very high-pitched"
        }
        
        # Maps rhythm density to descriptive terms
        self.rhythm_density_map = {
            (0, 0.5): "sparse",
            (0.5, 1.5): "moderate",
            (1.5, 2.5): "steady",
            (2.5, 3.5): "dense",
            (3.5, 5): "very dense"
        }
        
        # Maps average rhythm strength to tempo suggestions
        self.tempo_map = {
            (0, 1): "slow",
            (1, 2): "moderate",
            (2, 3): "upbeat",
            (3, 5): "fast"
        }
        
        # Timbral descriptors based on MFCC clusters
        self.timbre_descriptors = [
            "warm", "bright", "dark", "mellow", "rich", "thin", 
            "smooth", "rough", "airy", "breathy", "clear", "nasal", 
            "rounded", "piercing", "full", "hollow"
        ]
        
        # Style suggestions based on feature combinations
        self.style_map = {
            # (pitch_range, rhythm_density, timbre_index) -> style
            ("low-pitched", "sparse", 0): "ambient",
            ("low-pitched", "moderate", 0): "folk",
            ("low-pitched", "dense", 0): "blues",
            ("mid-range", "sparse", 0): "ballad",
            ("mid-range", "moderate", 0): "pop",
            ("mid-range", "dense", 0): "rock",
            ("high-pitched", "sparse", 0): "classical",
            ("high-pitched", "moderate", 0): "jazz",
            ("high-pitched", "dense", 0): "electronic"
        }
        
        # Add MusicGen-specific vocabulary
        self.musicgen_genres = [
            "ambient", "folk", "blues", "ballad", "pop", "rock", 
            "classical", "jazz", "electronic", "cinematic", "orchestral"
        ]
        
        self.musicgen_adjectives = [
            "melodic", "harmonic", "rhythmic", "emotional", "energetic",
            "calm", "uplifting", "dynamic", "atmospheric", "expressive"
        ]
        
        self.musicgen_instruments = [
            "flute", "guitar", "piano", "violin", "cello", "drums",
            "saxophone", "trumpet", "synthesizer", "voice", "orchestra"
        ]
        
        # Maximum token length for MusicGen prompt
        self.max_token_length = 512
        
        # Approximate token count per word (conservative estimate)
        self.token_ratio = 1.3
    
    def _get_range_descriptor(self, value: float, range_map: Dict[Tuple[float, float], str]) -> str:
        """Get descriptor for a value based on a range mapping"""
        for (low, high), descriptor in range_map.items():
            if low <= value < high:
                return descriptor
        return list(range_map.values())[0]  # Default to first value if not found
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a string based on word count"""
        words = text.split()
        return int(len(words) * self.token_ratio) + 1  # +1 for safety margin
    
    def _analyze_pitch_contour(self, pitch_tokens: List[int]) -> Dict[str, Any]:
        """Analyze pitch contour for melody characteristics"""
        if len(pitch_tokens) == 0 or all(p <= 0 for p in pitch_tokens):
            return {
                "range": "unknown",
                "avg_pitch": 60,
                "pitch_variability": "unknown",
                "contour": "unknown"
            }
        
        valid_pitches = [p for p in pitch_tokens if p > 0]
        
        if not valid_pitches:
            return {
                "range": "unknown",
                "avg_pitch": 60,
                "pitch_variability": "unknown",
                "contour": "unknown"
            }
        
        min_pitch = min(valid_pitches)
        max_pitch = max(valid_pitches)
        avg_pitch = sum(valid_pitches) / len(valid_pitches)
        
        # Calculate pitch range descriptor
        range_descriptor = self._get_range_descriptor(avg_pitch, self.pitch_range_map)
        
        # Calculate pitch variability
        pitch_variability = "stable"
        if max_pitch - min_pitch > 12:
            pitch_variability = "highly variable"
        elif max_pitch - min_pitch > 7:
            pitch_variability = "moderately variable"
        
        # Analyze contour
        contour = "fluctuating"
        if len(valid_pitches) >= 4:
            first_half = valid_pitches[:len(valid_pitches)//2]
            second_half = valid_pitches[len(valid_pitches)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg + 3:
                contour = "ascending"
            elif first_avg > second_avg + 3:
                contour = "descending"
            elif max(valid_pitches) - min(valid_pitches) < 5:
                contour = "fairly static"
        
        return {
            "range": range_descriptor,
            "avg_pitch": avg_pitch,
            "pitch_variability": pitch_variability,
            "contour": contour
        }
    
    def _analyze_rhythm(self, rhythm_tokens: List[int]) -> Dict[str, Any]:
        """Analyze rhythm tokens for rhythmic characteristics"""
        if len(rhythm_tokens) == 0:
            return {
                "density": "moderate",
                "avg_strength": 2,
                "pattern": "steady"
            }
        
        if not rhythm_tokens:
            return {
                "density": "moderate",
                "avg_strength": 2,
                "pattern": "steady"
            }
        
        avg_strength = sum(rhythm_tokens) / len(rhythm_tokens)
        
        # Calculate rhythm density
        density = self._get_range_descriptor(avg_strength, self.rhythm_density_map)
        
        # Analyze rhythmic pattern
        pattern = "steady"
        if len(rhythm_tokens) >= 4:
            differences = [abs(rhythm_tokens[i] - rhythm_tokens[i-1]) for i in range(1, len(rhythm_tokens))]
            avg_diff = sum(differences) / len(differences)
            
            if avg_diff > 2:
                pattern = "highly varied"
            elif avg_diff > 1:
                pattern = "varied"
            elif avg_diff < 0.5 and len(set(rhythm_tokens)) <= 2:
                pattern = "repetitive"
        
        # Detect accent patterns
        accents = []
        for i, strength in enumerate(rhythm_tokens):
            if strength >= 3:
                position = "downbeat" if i % 4 == 0 else "upbeat"
                accents.append(position)
        
        accent_pattern = "balanced"
        if len(accents) > 0:
            downbeats = accents.count("downbeat")
            upbeats = accents.count("upbeat")
            
            if downbeats > upbeats * 2:
                accent_pattern = "downbeat-heavy"
            elif upbeats > downbeats * 2:
                accent_pattern = "syncopated"
        
        return {
            "density": density,
            "avg_strength": avg_strength,
            "pattern": pattern,
            "accent_pattern": accent_pattern
        }
    
    def _analyze_timbre(self, mfcc_tokens: List[int]) -> Dict[str, Any]:
        """Analyze MFCC tokens for timbral characteristics"""
        if len(mfcc_tokens) == 0:  # Fixed: proper way to check if array is empty
            return {
                "primary_descriptor": "neutral",
                "consistency": "consistent"
            }
        
        # Count frequency of each cluster
        unique, counts = np.unique(mfcc_tokens, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        
        # Find most common clusters
        top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Map clusters to timbral descriptors
        primary_index = top_clusters[0][0] % len(self.timbre_descriptors)
        secondary_index = top_clusters[1][0] % len(self.timbre_descriptors) if len(top_clusters) > 1 else primary_index
        
        primary_descriptor = self.timbre_descriptors[primary_index]
        secondary_descriptor = self.timbre_descriptors[secondary_index]
        
        # Check timbre consistency
        consistency = "consistent"
        if len(top_clusters) >= 2:
            ratio = top_clusters[0][1] / top_clusters[1][1]
            if ratio < 1.5:
                consistency = "varied"
        
        return {
            "primary_descriptor": primary_descriptor,
            "secondary_descriptor": secondary_descriptor,
            "consistency": consistency
        }
    
    def generate_prompt(self, quantized: Dict[str, Any], instrument: str = "flute", duration: float = 10.0) -> str:
        """Generate a single MusicGen-optimized prompt from quantized features"""
        # Analyze different feature types
        pitch_analysis = self._analyze_pitch_contour(quantized['pitch_tokens'])
        rhythm_analysis = self._analyze_rhythm(quantized['rhythm_tokens'])
        timbre_analysis = self._analyze_timbre(quantized['mfcc_tokens'])
        
        # Determine musical attributes
        tempo = self._get_range_descriptor(rhythm_analysis['avg_strength'], self.tempo_map)
        
        # Create style description
        style_key = (
            pitch_analysis['range'].split('-')[0],  # Just use first part of range descriptor
            rhythm_analysis['density'],
            0  # Default index
        )
        style = self.style_map.get(style_key, "melodic")
        
        # Ensure instrument is just the name (e.g., "guitar"), not "solo guitar" yet
        clean_instrument = instrument.lower().replace("solo", "").strip()
        if clean_instrument not in [i.lower() for i in self.musicgen_instruments]:
            clean_instrument = "instrument" # Use a very generic term if unknown
        
        # Ensure style is in the known genres or use a fallback
        if style not in self.musicgen_genres:
            style = "melodic" if style not in self.musicgen_adjectives else style
        
        if duration <= 10.0:
            # Generate the enhanced 10-second prompt directly
            prompt_parts = [
                f"A clear, high-quality recording of a solo {clean_instrument}, exactly 10 seconds long.",
                f"This 10-second piece features the {clean_instrument} prominently.",
                f"The melody is {pitch_analysis['range']} and {pitch_analysis['contour']}.",
                f"It has a {rhythm_analysis['density']} rhythm.",
                random.choice(["The performance is expressive.", "Focus on the instrumental nuances.", "A short musical statement."]),
                f"Genre tags could include: solo instrumental, {clean_instrument}.",
                f"Primary instrument: {clean_instrument}. Duration: 10 seconds."
            ]
            final_prompt = " ".join(prompt_parts)
        else:
            # Generate a standard prompt for longer durations
            prompt_parts = [
                f"Solo {clean_instrument} performance.",
                f"The melody is {pitch_analysis['range']} and {pitch_analysis['contour']}.",
                f"Rhythm is {rhythm_analysis['density']}.",
                f"Overall feel: {tempo}.",
                f"Genre: solo instrumental, {clean_instrument}."
            ]
            final_prompt = " ".join(prompt_parts)
        
        return final_prompt


def map_features_to_prompt(quantized_features: Dict[str, Any], instrument: str = "flute", duration: float = 10.0) -> str:
    """Convert quantized features to a single MusicGen prompt"""
    generator = PromptGenerator()
    return generator.generate_prompt(quantized_features, instrument, duration)