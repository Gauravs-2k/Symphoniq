import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pathlib import Path
import scipy.io.wavfile
import numpy as np
import os
import argparse
import sys
import re
import platform

def get_optimal_device():
    """Detect the optimal device for PyTorch (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    # Check for Apple Silicon GPU (M1/M2/M3)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"

class InstrumentalGenerator:
    def __init__(self, model_name="facebook/musicgen-medium"):
        """Initialize MusicGen model using Transformers"""
        # Apply patches for safe token shifting
        import transformers.models.musicgen.modeling_musicgen as musicgen_module
        
        def safe_shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id=None):
            if decoder_start_token_id is None:
                decoder_start_token_id = 0
                
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
            shifted_input_ids[:, 0] = decoder_start_token_id
            
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
            
            return shifted_input_ids
        
        # Apply the patch
        musicgen_module.shift_tokens_right = safe_shift_tokens_right
        
        # Set up cache directory in the models folder
        self.cache_dir = Path(__file__).parent / "cached__medium_models"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Get the optimal device for current hardware
        self.device = get_optimal_device()
        system_info = platform.system()
        
        if self.device == "mps":
            print(f"Using Apple Silicon GPU on {system_info}")
        elif self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using NVIDIA GPU: {gpu_name} on {system_info}")
        else:
            print(f"Using CPU on {system_info}")
        
        print(f"Loading MusicGen model from local cache: {self.cache_dir}")
        
        try:
            # Try to load from cache
            config = MusicgenForConditionalGeneration.config_class.from_pretrained(
                self.cache_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            config.decoder_start_token_id = 0
            
            self.processor = AutoProcessor.from_pretrained(
                self.cache_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # Add torch_dtype for better performance on all hardware
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.cache_dir,
                config=config,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            print("Successfully loaded model from local cache")
            
        except Exception as e:
            print(f"Error loading local model: {str(e)}")
            
            # Try fallback paths
            for subfolder in ["best_model", "final_model"]:
                fallback_path = self.cache_dir / subfolder
                if fallback_path.exists():
                    try:
                        self.processor = AutoProcessor.from_pretrained(fallback_path, local_files_only=True)
                        self.model = MusicgenForConditionalGeneration.from_pretrained(
                            fallback_path, 
                            local_files_only=True,
                            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                            low_cpu_mem_usage=True
                        )
                        print(f"Loaded model from fallback path: {fallback_path}")
                        break
                    except Exception as fallback_e:
                        print(f"Error loading from {fallback_path}: {str(fallback_e)}")
            
            # If still not loaded, use the pretrained model
            if not hasattr(self, 'model'):
                print(f"Falling back to HuggingFace model: {model_name}")
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    low_cpu_mem_usage=True
                )
        
        # Move to detected device
        self.model.to(self.device)
        print(f"Model loaded on {self.device} device")
        
        # Print memory usage if GPU
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)    # Convert to GB
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"GPU memory reserved: {memory_reserved:.2f} GB")
        
        # After loading the model, determine maximum prompt length
        try:
            # Get max length from config
            if hasattr(self.model.config, 'max_text_length'):
                self.max_prompt_length = self.model.config.max_text_length
            else:
                # Default for MusicGen models if not specified in config
                self.max_prompt_length = 512
                
            print(f"Maximum prompt length: {self.max_prompt_length} tokens")
        except Exception as e:
            print(f"Warning: Could not determine max prompt length: {str(e)}")
            # Set a safe default
            self.max_prompt_length = 512
    
    def generate_instrumental(self, prompt, output_path=None, duration=10, guidance_scale=3.0):
        """Generate instrumental audio from a prompt"""
        # Check prompt length
        encoded_prompt = self.processor.tokenizer(prompt, return_tensors="pt")
        prompt_length = encoded_prompt.input_ids.shape[1]
        
        if prompt_length > self.max_prompt_length:
            original_length = prompt_length
            # Truncate the prompt
            truncated_prompt = self.processor.tokenizer.decode(
                encoded_prompt.input_ids[0, :self.max_prompt_length], 
                skip_special_tokens=True
            )
            print(f"Warning: Prompt exceeds maximum length ({original_length} > {self.max_prompt_length} tokens)")
            print(f"Truncating prompt to {self.max_prompt_length} tokens")
            prompt = truncated_prompt
        else:
            print(f"Prompt length: {prompt_length}/{self.max_prompt_length} tokens")
        
        print(f"Generating instrumental with prompt: {prompt[:100]}...")
        
        # Process the text prompt
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Calculate max tokens based on duration (32000 samples per second, each token ~50 ms)
        max_tokens = int((duration * 32000) / 960)  # Each token is about 960 samples
        print(f"Generating {max_tokens} tokens for {duration}s of audio")
        
        # Generate audio
        with torch.no_grad():
            generated_audio = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                guidance_scale=guidance_scale,
                temperature=0.7,
            )
        
        # Convert to numpy and scale
        audio_data = generated_audio.cpu().numpy().squeeze()
        
        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Scale to 16-bit int range
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV
            scipy.io.wavfile.write(output_path, rate=32000, data=audio_int16)
            print(f"Saved audio to {output_path}")
            
            return output_path
        
        return audio_data  # Return the audio data if no output path

# Add this function to process prompt files
def generate_from_vocal_prompt(prompt_file, output_path=None, instrument="flute", duration=20):
    """Generate instrumental music from a vocal prompt file"""
    # Check if file exists
    if not os.path.exists(prompt_file):
        print(f"Error: Prompt file not found: {prompt_file}")
        return None
    
    # Read prompt from file
    with open(prompt_file, 'r') as f:
        content = f.read()
    
    # Try to extract the detailed prompt if it exists
    detailed_match = re.search(r'Detailed Prompt:(.*?)(?=\n\n|\Z)', content, re.DOTALL)
    if detailed_match:
        prompt = detailed_match.group(1).strip()
    else:
        # Try to extract summary prompt
        summary_match = re.search(r'Summary Prompt:(.*?)(?=\n\n|\Z)', content, re.DOTALL)
        if summary_match:
            prompt = summary_match.group(1).strip()
        else:
            # Use the whole file if no specific format is found
            prompt = content.strip()
    
    print(f"Using prompt:\n{prompt[:200]}...")
    
    # Set default output path if none provided
    if output_path is None:
        output_path = Path(prompt_file).with_suffix('').with_name(f"{Path(prompt_file).stem}_{instrument}.wav")
    
    # Initialize generator
    generator = InstrumentalGenerator()
    
    # Generate instrumental audio
    result = generator.generate_instrumental(
        prompt=prompt,
        output_path=output_path,
        duration=duration
    )
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instrumental music from vocal prompts")
    parser.add_argument("prompt_file", help="Path to the prompt file")
    parser.add_argument("-o", "--output", help="Path to save the output audio file")
    parser.add_argument("-i", "--instrument", default="flute", 
                       help="Target instrument (default: flute)")
    parser.add_argument("-d", "--duration", type=int, default=20,
                       help="Duration in seconds (default: 20)")
    
    args = parser.parse_args()
    
    result = generate_from_vocal_prompt(
        args.prompt_file, 
        args.output, 
        args.instrument,
        args.duration
    )
    
    if result:
        print(f"Successfully generated audio: {result}")
    else:
        print("Failed to generate audio")
        sys.exit(1)
