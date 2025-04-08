from typing import Dict, Any, List, Tuple
import numpy as np
import librosa

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
    
    def _create_optimized_prompt(self, pitch_analysis, rhythm_analysis, 
                                timbre_analysis, tempo, style, instrument) -> str:
        """Create a concise prompt optimized for MusicGen's token limit"""
        # Core musical attributes - most important information first
        core_prompt = f"{tempo} {style} {instrument}. "
        
        # Add pitch information if we have room
        pitch_info = f"{pitch_analysis['range']} {pitch_analysis['contour']} melody. "
        
        # Add rhythm information if we have room
        rhythm_info = f"{rhythm_analysis['density']} rhythm. "
        
        # Add timbre information if we have room
        timbre_info = f"{timbre_analysis['primary_descriptor']} tone. "
        
        # Standard quality suffix
        quality = "Clear professional recording."
        
        # Combine parts, checking token length
        prompt = core_prompt
        
        if self._estimate_token_count(prompt + pitch_info) < self.max_token_length:
            prompt += pitch_info
            
            if self._estimate_token_count(prompt + rhythm_info) < self.max_token_length:
                prompt += rhythm_info
                
                if self._estimate_token_count(prompt + timbre_info) < self.max_token_length:
                    prompt += timbre_info
                    
                    if self._estimate_token_count(prompt + quality) < self.max_token_length:
                        prompt += quality
        
        return prompt
    
    def _create_enhanced_10sec_prompt(self, pitch_analysis, rhythm_analysis, 
                                     timbre_analysis, tempo, style, instrument) -> str:
        """Create a highly detailed prompt utilizing more of the available token limit"""
        # Create a comprehensive, detailed prompt that uses more of the 512 token limit
        prompt = f"Create a captivating {tempo} {style} composition featuring solo {instrument}. "
        
        # Detailed melody section
        prompt += f"The melody should be primarily {pitch_analysis['range']}, moving in a {pitch_analysis['contour']} direction "
        if pitch_analysis['pitch_variability'] != "unknown":
            prompt += f"with {pitch_analysis['pitch_variability']} note choices that create musical interest. "
        else:
            prompt += "with expressive phrasing and articulation. "
        
        # Add note range details
        if pitch_analysis['range'] == "low-pitched":
            prompt += f"Focus on the lower register of the {instrument}, creating a rich foundation. "
        elif pitch_analysis['range'] == "mid-range":
            prompt += f"Utilize the middle register of the {instrument} for a balanced, resonant sound. "
        elif pitch_analysis['range'] == "high-pitched":
            prompt += f"Explore the upper register of the {instrument}, achieving brightness and projection. "
        
        # Detailed rhythm section
        prompt += f"The rhythm should be {rhythm_analysis['density']} and {rhythm_analysis['pattern']}, "
        prompt += f"creating a {tempo} pulse that drives the piece forward. "
        
        # Add rhythm specifics
        if rhythm_analysis['accent_pattern'] != "balanced":
            prompt += f"Incorporate {rhythm_analysis['accent_pattern']} accents to create rhythmic tension and release. "
        else:
            prompt += "Maintain a naturally flowing, balanced accent pattern throughout. "
        
        # Add style-specific techniques
        if style == "jazz":
            prompt += "Include subtle jazz-inspired phrasings with appropriate swing feel. "
        elif style == "classical":
            prompt += "Maintain classical articulation and phrasing with careful attention to dynamics. "
        elif style == "folk":
            prompt += "Incorporate folk-inspired ornaments and simple, memorable motifs. "
        elif style == "rock":
            prompt += "Add occasional rhythmic intensity and drive characteristic of rock music. "
        
        # Detailed timbre description
        prompt += f"The tone should be predominantly {timbre_analysis['primary_descriptor']} and {timbre_analysis['secondary_descriptor']}, "
        prompt += f"with a {timbre_analysis['consistency']} timbral quality throughout the performance. "
        
        # Tone production guidance
        if instrument == "guitar":
            prompt += "Play with a mix of fingerpicking and strumming techniques as appropriate. "
        elif instrument == "piano":
            prompt += "Utilize sensitive pedaling to blend harmonies while maintaining clarity. "
        elif instrument == "violin" or instrument == "cello":
            prompt += "Employ a variety of bowing techniques to achieve expressive phrasing. "
        elif instrument == "flute" or instrument == "saxophone":
            prompt += "Use controlled breathing and articulation for a polished sound. "
        
        # Performance and expression
        prompt += f"The performance should convey emotional {style} sensibilities with natural dynamics, "
        prompt += "starting softer and building in intensity where musically appropriate. "
        
        # Production quality
        prompt += "The recording should have professional studio quality with natural room acoustics, "
        prompt += "clear definition, balanced frequency response, and subtle reverb that enhances the instrument's resonance. "
        
        # Final stylistic instruction
        prompt += f"Overall, create an engaging {style} piece that showcases the expressive capabilities of the {instrument}."
        
        return prompt

    def generate_prompt(self, quantized: Dict[str, Any], instrument: str = "flute") -> Dict[str, str]:
        """Generate a MusicGen-optimized prompt from quantized features"""
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
        
        # Ensure instrument is in the known list or default to a generic term
        if instrument.lower() not in [i.lower() for i in self.musicgen_instruments]:
            instrument = "solo instrument"
        
        # Ensure style is in the known genres or use a fallback
        if style not in self.musicgen_genres:
            style = "melodic" if style not in self.musicgen_adjectives else style
        
        # Create an optimized prompt that fits within token limit
        musicgen_prompt = self._create_optimized_prompt(
            pitch_analysis, rhythm_analysis, timbre_analysis, tempo, style, instrument
        )
        
        # Enhanced prompt specifically designed for 10-second audio clips
        enhanced_10sec_prompt = self._create_enhanced_10sec_prompt(
            pitch_analysis, rhythm_analysis, timbre_analysis, tempo, style, instrument
        )
        
        # Legacy compact format (used as fallback)
        compact_prompt = f"{tempo} {style} {instrument}. {pitch_analysis['range']}. {timbre_analysis['primary_descriptor']}."
        
        # Ultra-compact for very short clips
        minimal_prompt = f"{style} {instrument}."
        
        # Create all prompt versions for the return value
        return {
            "musicgen_prompt": musicgen_prompt,            # Primary optimized prompt
            "enhanced_10sec_prompt": enhanced_10sec_prompt, # Detailed prompt for 10-sec clips
            "compact_prompt": compact_prompt,              # Legacy compact version
            "minimal_prompt": minimal_prompt,              # Minimal version
            "detailed_prompt": self._generate_legacy_detail(pitch_analysis, rhythm_analysis, timbre_analysis, tempo, style, instrument),
            "keyword_prompt": f"{style}. {tempo}. {instrument}.",
            "legacy_full_prompt": self._generate_legacy_full_prompt(pitch_analysis, rhythm_analysis, timbre_analysis, tempo, style, instrument),
            "legacy_summary": self._generate_legacy_summary(pitch_analysis, rhythm_analysis, timbre_analysis, tempo, style, instrument)
        }
    
    def _generate_legacy_full_prompt(self, pitch_analysis, rhythm_analysis, 
                                    timbre_analysis, tempo, style, instrument):
        """Generate the original detailed prompt format for backward compatibility"""
        intro = f"Create a {instrument} solo with the following characteristics:"
        
        melody_desc = (
            f"A {pitch_analysis['range']} melody with a {pitch_analysis['contour']} contour, "
            f"featuring {pitch_analysis['pitch_variability']} note choices."
        )
        
        rhythm_desc = (
            f"The piece has a {rhythm_analysis['density']} rhythm with a {rhythm_analysis['pattern']} pattern, "
            f"and a {rhythm_analysis['accent_pattern']} accent structure at a {tempo} tempo."
        )
        
        timbre_desc = (
            f"The tone should be {timbre_analysis['primary_descriptor']} and {timbre_analysis['secondary_descriptor']}, "
            f"with a {timbre_analysis['consistency']} character throughout."
        )
        
        style_desc = (
            f"Overall, aim for a {style} feel, with expressive phrasing and natural dynamics. "
            f"The sound should be clear and professionally recorded with subtle reverb."
        )
        
        return f"{intro}\n\n{melody_desc}\n\n{rhythm_desc}\n\n{timbre_desc}\n\n{style_desc}"
    
    def _generate_legacy_summary(self, pitch_analysis, rhythm_analysis, 
                               timbre_analysis, tempo, style, instrument):
        """Generate the original summary prompt for backward compatibility"""
        return (
            f"Create a {tempo} {style} {instrument} solo with a {pitch_analysis['range']} "
            f"{pitch_analysis['contour']} melody, {rhythm_analysis['density']} rhythm, and "
            f"{timbre_analysis['primary_descriptor']} tone."
        )
    
    def _generate_legacy_detail(self, pitch_analysis, rhythm_analysis, timbre_analysis, tempo, style, instrument):
        """Generate a compact but detailed prompt"""
        return (
            f"Generate a {tempo} {style} {instrument} piece. "
            f"{pitch_analysis['range']} {pitch_analysis['contour']} melody, "
            f"{rhythm_analysis['density']} rhythm, {timbre_analysis['primary_descriptor']} tone."
        )


def map_features_to_prompt(quantized_features: Dict[str, Any], instrument: str = "flute") -> Dict[str, str]:
    """Convert quantized features to MusicGen prompts"""
    generator = PromptGenerator()
    return generator.generate_prompt(quantized_features, instrument)