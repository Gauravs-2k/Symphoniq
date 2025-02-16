import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pathlib import Path
import scipy.io.wavfile
import numpy as np
import os
import pretty_midi
from ..processors.validate_midi import validate_midi_file
import sys


sys.path.append(str(Path(__file__).parent.parent.parent))

from src.processors.validate_midi import validate_midi_file

class InstrumentalGenerator:
    def __init__(self, model_name="facebook/musicgen-small"):
        """Initialize MusicGen model using Transformers"""
        # Set up cache directory in the models folder
        self.cache_dir = Path(__file__).parent / "cached_models"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Loading MusicGen model from local cache: {self.cache_dir}")
        
        try:
            # Try to load model from local cache only
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=True  # Only use local files
            )
            
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=True  # Only use local files
            )
            print("Successfully loaded model from local cache")
            
        except Exception as e:
            print(f"Error loading local model: {str(e)}")
            print("\nPlease run the following commands to download the model first:")
            print("\npython")
            print("from transformers import AutoProcessor, MusicgenForConditionalGeneration")
            print(f"processor = AutoProcessor.from_pretrained('{model_name}')")
            print(f"model = MusicgenForConditionalGeneration.from_pretrained('{model_name}')")
            print(f"processor.save_pretrained('{self.cache_dir}')")
            print(f"model.save_pretrained('{self.cache_dir}')")
            raise Exception("Local model not found. Please download it first.")
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
    
    def generate_instrumental(self, prompt: str, output_path: Path, duration: int = 30) -> str:
        """
        Generate instrumental music from text prompt
        Args:
            prompt (str): Text description of desired music
            output_path (Path): Path to save generated audio
            duration (int): Duration in seconds (default: 30)
        Returns:
            str: Path to generated audio file
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process the text prompt
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            print(f"Generating {duration} seconds of audio...")
            
            # Set optimal parameters for 30-second generation
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=1500,  # 30 seconds * 50 tokens/second
                guidance_scale=3.5,   # Increased for better prompt following
                temperature=0.8,      # Slightly increased for more variation
                do_sample=True,       # Enable sampling
                use_cache=True        # Enable caching for faster generation
            )
            
            # Convert to numpy array
            audio_data = audio_values.cpu().numpy().squeeze()
            
            # Save as WAV file (sample rate is 32000 for MusicGen)
            scipy.io.wavfile.write(
                output_path,
                rate=32000,
                data=(audio_data * 32767).astype(np.int16)
            )
            
            print(f"Generated audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error generating instrumental: {str(e)}")
            return None
        
    def generate_from_midi(self, midi_path: Path, output_path: Path, start_time: float = 0, duration: int = 300) -> str:
        """
        Generate instrumental music based on MIDI file
        Args:
            midi_path (Path): Path to input MIDI file
            output_path (Path): Path to save generated audio
            start_time (float): Start time in seconds for processing chunk
            duration (int): Duration in seconds to generate
        Returns:
            str: Path to generated audio file
        """
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Extract features from the specified time chunk
            tempo = midi_data.get_tempo_changes()[1][0] if midi_data.get_tempo_changes()[1].size > 0 else 120
            
            # Get instruments for this segment
            instruments = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    # Filter notes in the current time window
                    notes_in_range = [note for note in instrument.notes 
                                    if note.start >= start_time and note.start < start_time + duration]
                    if notes_in_range:
                        instruments.append(pretty_midi.INSTRUMENT_MAP[instrument.program])
            
            if not instruments:
                instruments = ["piano"]  # Default if no instruments found in chunk
            
            # Create prompt for this chunk
            tempo_desc = "fast" if tempo > 100 else "slow"
            prompt = (f"Generate a {tempo_desc} tempo instrumental piece using "
                     f"{', '.join(set(instruments))}. "
                     f"The piece should be {duration} seconds long.")
            
            return self.generate_instrumental(prompt, output_path, duration=duration)
            
        except Exception as e:
            print(f"Error processing MIDI file: {str(e)}")
            return None