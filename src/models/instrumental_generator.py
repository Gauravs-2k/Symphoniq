import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pathlib import Path
import scipy.io.wavfile
import numpy as np
import os
import pretty_midi
import sys


sys.path.append(str(Path(__file__).parent.parent.parent))

from src.processors.validate_midi import validate_midi_file

class InstrumentalGenerator:
    def __init__(self, model_name="facebook/musicgen-small"):
        """Initialize MusicGen model using Transformers"""
        # Set up cache directory in the models folder
        # model_path = Path(__file__).parent / "fine_tuned_model" / "final_model"

        self.cache_dir = Path(__file__).parent / "fine_tuned_model" / "final_model"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Loading MusicGen model from local cache: {self.cache_dir}")
        
        try:
            # Try to load model directly from the cache directory instead of using model_name
            self.processor = AutoProcessor.from_pretrained(
                self.cache_dir,  # Load directly from cache path
                local_files_only=True
            )
            
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.cache_dir,  # Load directly from cache path
                local_files_only=True
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
            max_tokens = int(duration * 50)
            # Set optimal parameters for 30-second generation
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # 30 seconds * 50 tokens/second
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
    
    
    def generate_from_midi(self, midi_path: Path, output_path: Path,instrument="flute", start_time: float = 0, duration: int = 100) -> str:
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
            
            # Extract melody information
            notes = []
            for instr in midi_data.instruments:
                if not instr.is_drum:
                    notes.extend(instr.notes)
            
            # Calculate melody density
            if notes:
                # Get start time by finding earliest note start time (or use 0)
                start_time = min([note.start for note in notes]) if notes else 0
                notes_per_second = len(notes) / (midi_data.get_end_time() - start_time)
                density = "complex" if notes_per_second > 4 else "simple"
            else:
                density = "simple"
            techniques = {
                "flute": "breath control, vibrato, trills, and flutter tonguing",
                "violin": "vibrato, pizzicato, staccato, and legato bowing",
                "guitar": "fingerpicking, strumming, hammer-ons, and pull-offs",
                "piano": "pedaling, arpeggios, glissando, and dynamic control",
                "saxophone": "vibrato, growling, bending, and subtone",
                "trumpet": "tonguing, vibrato, glissando, and mutes",
                "cello": "vibrato, pizzicato, sul ponticello, and col legno",
                "harp": "glissandos, harmonics, and près de la table"
            }
            instrument_descriptions = {
                "flute": "a wooden flute or Native American flute with warmth and clarity",
                "violin": "a classical violin with emotional expression and rich tone",
                "guitar": "an acoustic guitar with clean fingerpicking style",
                "piano": "a grand piano with delicate touch and expressive dynamics",
                "saxophone": "a smooth jazz saxophone with soulful expression",
                "trumpet": "a bright brass trumpet with clear articulation",
                "cello": "a deep and resonant cello with warm tones",
                "harp": "a ethereal harp with delicate plucking and resonance"
            }
            
            instrument_desc = instrument_descriptions.get(
                instrument.lower(), 
                f"a beautiful {instrument}"
            )
            # Create prompt specifically for flute
            tempo_desc = "fast" if tempo > 100 else "slow"
            prompt = (
                f"Generate a high-quality {tempo_desc} tempo {instrument} solo with a {density} melody. "
                f"The {instrument} should have clear articulation, proper dynamics, and natural phrasing. "
                f"Include authentic {instrument} techniques like {techniques.get(instrument.lower(), "expressive playing and dynamics")}. "
                f"The piece should sound like {instrument_desc} with pristine audio quality, "
                f"recorded in a concert hall with natural reverb. The performance should be "
                f"expressive with subtle vibrato and dynamic contrast."
            )
            print(f"Generating with prompt: {prompt}")
            return self.generate_instrumental(prompt, output_path, duration=duration)
            
        except Exception as e:
            print(f"Error processing MIDI file: {str(e)}")
            return None


if __name__ == "__main__":
    import argparse
    # Option 1: Use hardcoded parameters - just uncomment these lines to run directly
    midi_file_path = "/Users/gauravs/Documents/Symphoniq/src/data/input/midi/song_10_vocals.mid"
    output_file_path = "/Users/gauravs/Documents/Symphoniq/src/data/output/flute_output.wav"
    
    try:
        print("Initializing InstrumentalGenerator...")
        generator = InstrumentalGenerator()
        
        print(f"Generating flute sound from MIDI: {midi_file_path}")
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Generate flute sound from MIDI
        result = generator.generate_from_midi(
            midi_path=Path(midi_file_path),
            output_path=Path(output_file_path),
            instrument = "flute",
            duration=10
        )
        if result:
            print(f"Generated audio saved to: {output_file_path}")
        else:
            print("Failed to generate audio")
    except Exception as e:
        print(f"Error: {str(e)}")
    