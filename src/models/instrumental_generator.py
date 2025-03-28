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
        self.cache_dir = Path(__file__).parent / "cached__medium_models"
        # self.cache_dir = Path(__file__).parent / "flute_model" / "final_model"
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
    
    
    
    def generate_instrumental(self, prompt: str, output_path: Path, duration: int = 30, 
                             guidance_scale=3.5, temperature=0.7) -> str:
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
            
            # Use the parameters passed to the function
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
                do_sample=True,
                use_cache=True
            )
            
            # Convert to numpy array
            audio_data = audio_values.cpu().numpy().squeeze()
            
            # Save as WAV file (sample rate is 32000 for MusicGen)
            scipy.io.wavfile.write(
                output_path,
                rate=32000,
                data=(audio_data * 32767).astype(np.int16)
            )
            
            print(f"Generated instrument audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error generating instrumental: {str(e)}")
            return None
    
        
    def generate_from_midi(self, midi_path: Path, output_path: Path, instrument="flute", start_time: float = 0, duration: int = 30) -> str:
        """Generate instrumental music based on MIDI file"""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Extract key signature, tempo, time signature
            tempo = 120  # Default
            try:
                _, tempos = midi_data.get_tempo_changes()
                if len(tempos) > 0:
                    tempo = int(tempos[0])
            except:
                pass
                
            # Create highly focused, stabilizing prompt
            prompt = (
                f"Create a realistic solo flute performance of this melody. "
                f"The flute should play at tempo {tempo} BPM with precise articulation and clean attacks. "
                f"The flute sound should have a pure, airy tone characteristic of a professional flutist. "
                f"Use moderate vibrato only on longer notes. Pay special attention to the breath control, "
                f"with appropriate phrasing and natural breathing points. "
                f"The timbre should be bright and clear in the upper register and warm and rich in the middle register. "
                f"Ensure the performance has expressive dynamics and subtle ornamentations typical of classical flute playing."
            )
            
            print(f"Generating with prompt: {prompt}")
            
            # Adjust generation parameters
            return self.generate_instrumental(
                prompt, 
                output_path, 
                duration=duration,
                guidance_scale=3.5,  # Stronger prompt adherence
                temperature=0.7      # More predictable output
            )
                
        except Exception as e:
            print(f"Error processing MIDI file: {str(e)}")
            return None


if __name__ == "__main__":
    import argparse
    
    # Use paths relative to project root
    midi_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "data", "input", "midi", "song_10_vocals.mid")
    output_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   "data", "output", "flute_output.wav")
    
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
            instrument="flute",
            duration=10
        )
        if result:
            print(f"Generated audio saved to: {output_file_path}")
        else:
            print("Failed to generate audio")
    except Exception as e:
        print(f"Error: {str(e)}")
    