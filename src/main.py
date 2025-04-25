from converters.mp3_to_wav import convert_to_wav
from converters.wav_to_mp3 import convert_to_mp3
from processors.vocal_seperator import separate_audio
from processors.audio_merger import merge_audio
from models.instrumental_generator import InstrumentalGenerator
from models.prompt.feature_extraction import extract_features_from_file
from models.prompt.quantization import quantize_features
from models.prompt.dict_map import map_features_to_prompt
from processors.audio_chunk_dechunk import split_audio_into_chunks, merge_audio_from_paths
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pathlib import Path

def create_audio_comparison_plots(vocals_path, instrument_path, output_dir, base_name):
    """
    Create and save comparison plots between vocal and instrumental audio files
    
    Args:
        vocals_path (str): Path to vocals WAV file
        instrument_path (str): Path to generated instrument WAV file
        output_dir (str): Directory to save plots
        base_name (str): Base filename for the plots
    
    Returns:
        str: Path to saved plot image
    """
    plt.figure(figsize=(14, 10))
    
    # Load audio files
    vocals, sr_vocals = librosa.load(vocals_path, sr=None)
    instrument, sr_instrument = librosa.load(instrument_path, sr=None)
    
    # Plot waveforms
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(vocals, sr=sr_vocals, alpha=0.6)
    plt.title('Vocals Waveform')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 2)
    librosa.display.waveshow(instrument, sr=sr_instrument, alpha=0.6)
    plt.title('Generated Instrumental Waveform')
    plt.ylabel('Amplitude')
    
    # Plot spectrograms
    plt.subplot(4, 1, 3)
    vocals_spec = librosa.amplitude_to_db(np.abs(librosa.stft(vocals)), ref=np.max)
    librosa.display.specshow(vocals_spec, sr=sr_vocals, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Vocals Spectrogram')
    
    plt.subplot(4, 1, 4)
    instrument_spec = librosa.amplitude_to_db(np.abs(librosa.stft(instrument)), ref=np.max)
    librosa.display.specshow(instrument_spec, sr=sr_instrument, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Generated Instrumental Spectrogram')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{base_name}_audio_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Audio comparison plot saved to: {plot_path}")
    return plot_path

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

    # Step 3:  Convert the separated files into 10 seconds chunks
    success, result = split_audio_into_chunks(vocals_path, 10)
    if not success:
        print(f"Error: {result}")
        return
    
    generated_chunks = []
    for i in range(1, result + 1):
        chunk_file = os.path.join(os.path.dirname(vocals_path), f"output_chunks/chunk_{i}.wav")
        print(f"Reading chunk {i}: {chunk_file}")

        # Step 4: Generate prompt from vocals for MusicGen
        print("Generating MusicGen prompt from vocals...")
        try:
            # Extract audio feature

            features = extract_features_from_file(chunk_file)
            
            # Quantize features to token sequences
            quantized = quantize_features(features)
            
            # Generate MusicGen prompt
            prompt_to_use = map_features_to_prompt(quantized, instrument)
            
            # Save prompt to file
            prompt_file = os.path.join(os.path.dirname(vocals_path), f"{base_name}_chunk_{i}_prompt.txt")
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
            
        # Step 5: Generate instrument sound directly from prompt using InstrumentalGenerator
        print(f"Initializing InstrumentalGenerator for {instrument}...")
        generator = InstrumentalGenerator()
        
        generated_dir = os.path.join(os.path.dirname(wav_output), "generated")
        os.makedirs(generated_dir, exist_ok=True)
        instrument_output = os.path.join(generated_dir, f"{base_name}_{instrument}_chunk_{i}.wav")
        
        print(f"Generating {instrument} sound with prompt guidance...")
        # Use generate_instrumental instead of generate_from_midi
        instrument_path = generator.generate_instrumental(
            prompt=prompt_to_use,
            output_path=instrument_output,
            duration=20  # Adjust duration as needed
        )
        
        if not instrument_path:
            print(f"Error: {instrument} sound generation failed")
            return
            
        # Store the generated chunk path
        generated_chunks.append(instrument_path)
        
        # Validation using the comparison plot
        plots_dir = os.path.join(os.path.dirname(wav_output), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        comparison_plot = create_audio_comparison_plots(
            chunk_file, 
            instrument_path, 
            plots_dir, 
            f"{base_name}_chunk_{i}"
        )
        
        print(f"Processing complete for chunk {i}! Comparison plot: {comparison_plot}")
        print(f"Generated chunks: {generated_chunks}")

    # Step 6: Merge all generated instrument sounds with original instrumental
    final_dir = os.path.join(os.path.dirname(wav_output), "final")
    os.makedirs(final_dir, exist_ok=True)
    
    print("MERGING CHUNKS", generated_chunks)
    # First merge all generated chunks
    merged_chunks = os.path.join(final_dir, f"{base_name}_{instrument}_all_chunks.wav")
    if not merge_audio_from_paths(generated_chunks, merged_chunks):
        print("Error: Merging generated chunks failed")
        return
    
    # Then merge with original instrumental
    final_wav = os.path.join(final_dir, f"{base_name}_final.wav")
    if not merge_audio(instrumental_path, merged_chunks, final_wav):
        print("Error: Final audio merging failed")
        return
    
    # Optional: Convert final WAV to MP3
    final_mp3 = os.path.join(final_dir, f"{base_name}_final.mp3")
    if not convert_to_mp3(final_wav, final_mp3):
        print("Error: MP3 conversion failed")
        return
        
    print(f"Processing complete! Final files in: {final_dir}")
    return final_dir

if __name__ == "__main__":
    # Path to your MP3 file
    input_file = os.path.join(os.path.dirname(__file__), "data", "input", "song_10.mp3")
    
    # Specify the target instrument (default: guitar)
    target_instrument = "piano"
    
    process_audio(input_file, target_instrument)