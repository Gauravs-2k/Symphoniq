from converters.mp3_to_wav import convert_to_wav
from converters.wav_to_mp3 import convert_to_mp3
from processors.vocal_seperator import separate_audio
from processors.audio_merger import merge_audio
from models.instrumental_generator import InstrumentalGenerator
from models.prompt.feature_extraction import extract_features_from_file
from models.prompt.quantization import quantize_features
from models.prompt.dict_map import map_features_to_prompt
import os
from pathlib import Path

def process_audio(input_mp3_path, instrument="guitar"):
    """
    Main function to process audio file
    Args:
        input_mp3_path (str): Path to input MP3 file
        instrument (str): Target instrument for generation
    """
    # Convert relative path to absolute path
    input_mp3_path = os.path.abspath(input_mp3_path)
    
    # Validate input file exists
    if not os.path.exists(input_mp3_path):
        print(f"Error: Input file not found at {input_mp3_path}")
        return
    
    # Step 1: Convert MP3 to WAV
    wav_output = convert_to_wav(input_mp3_path)
    if not wav_output:
        print("Error: MP3 to WAV conversion failed")
        return

    print(f"Successfully converted {input_mp3_path} to WAV format")
    
    # Step 2: Separate vocals and instruments
    output_dir = os.path.join(os.path.dirname(wav_output), "separated")
    os.makedirs(output_dir, exist_ok=True)
    if not separate_audio(wav_output, output_dir):
        print("Error: Audio separation failed")
        return
    
    # Get paths for separated files
    base_name = os.path.splitext(os.path.basename(wav_output))[0]
    vocals_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
    instrumental_path = os.path.join(output_dir, f"{base_name}_other.wav")
    
    # Step 3: Generate prompt from vocals for MusicGen
    print("Generating MusicGen prompt from vocals...")
    try:
        # Extract audio features
        features = extract_features_from_file(vocals_path)
        
        # Quantize features to token sequences
        quantized = quantize_features(features)
        
        # Generate MusicGen prompt
        prompts = map_features_to_prompt(quantized, instrument)
        
        # Determine which prompt to use (enhanced for 10-sec clips)
        duration = len(features['audio_data']) / features['sample_rate']
        prompt_to_use = prompts['enhanced_10sec_prompt'] if duration <= 10 else prompts['musicgen_prompt']
        
        # Save prompt to file
        prompt_file = os.path.join(os.path.dirname(vocals_path), f"{base_name}_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt_to_use)
            
        print(f"Generated prompt: '{prompt_to_use}'")
        print(f"Prompt saved to: {prompt_file}")
        
        # Estimate token count
        token_count = len(prompt_to_use.split()) * 1.3
        print(f"Token count (estimate): {token_count:.0f}/512")
        
    except Exception as e:
        print(f"Warning: Prompt generation failed: {str(e)}")
        prompt_to_use = f"{instrument} solo in the style of the vocal melody."
        print(f"Using fallback prompt: '{prompt_to_use}'")
    
    # Step 4: Generate instrument sound directly from prompt using InstrumentalGenerator
    print(f"Initializing InstrumentalGenerator for {instrument}...")
    generator = InstrumentalGenerator()
    
    generated_dir = os.path.join(os.path.dirname(wav_output), "generated")
    os.makedirs(generated_dir, exist_ok=True)
    instrument_output = os.path.join(generated_dir, f"{base_name}_{instrument}.wav")
    
    print(f"Generating {instrument} sound with prompt guidance...")
    # Use generate_instrumental instead of generate_from_midi
    instrument_path = generator.generate_instrumental(
        prompt=prompt_to_use,
        output_path=instrument_output,
        duration=10  # Adjust duration as needed
    )
    
    if not instrument_path:
        print(f"Error: {instrument} sound generation failed")
        return
        
    # Step 5: Merge generated instrument sound with original instrumental
    final_dir = os.path.join(os.path.dirname(wav_output), "final")
    os.makedirs(final_dir, exist_ok=True)
    merged_wav = os.path.join(final_dir, f"{base_name}_final.wav")
    if not merge_audio(instrumental_path, instrument_path, merged_wav):
        print("Error: Audio merging failed")
        return
    
    # Optional: Convert final WAV to MP3
    final_mp3 = os.path.join(final_dir, f"{base_name}_final.mp3")
    if not convert_to_mp3(merged_wav, final_mp3):
        print("Error: MP3 conversion failed")
        return
        
    print(f"Processing complete! Final files in: {final_dir}")
    return final_dir

if __name__ == "__main__":
    # Path to your MP3 file
    input_file = os.path.join(os.path.dirname(__file__), "data", "input", "eterna-cancao-wav-12569.mp3")
    
    # Specify the target instrument (default: guitar)
    target_instrument = "guitar"
    
    process_audio(input_file, target_instrument)