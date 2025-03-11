from converters.mp3_to_wav import convert_to_wav
from converters.wav_to_mp3 import convert_to_mp3
from processors.vocal_seperator import separate_audio
from processors.midi_converter_new import vocal_to_midi
from processors.audio_merger import merge_audio
from models.instrumental_generator import InstrumentalGenerator
import os
from pathlib import Path

def process_audio(input_mp3_path):
    """
    Main function to process audio file
    Args:
        input_mp3_path (str): Path to input MP3 file
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
    
    # Step 3: Convert vocals to MIDI
    midi_dir = os.path.join(os.path.dirname(wav_output), "midi")
    os.makedirs(midi_dir, exist_ok=True)
    midi_output = os.path.join(midi_dir, f"{base_name}_vocals.mid")
    if not vocal_to_midi(vocals_path, midi_output):
        print("Error: MIDI conversion failed")
        return
    
    # Step 4: Generate flute sound from MIDI using InstrumentalGenerator
    print("Initializing InstrumentalGenerator...")
    generator = InstrumentalGenerator()
    
    generated_dir = os.path.join(os.path.dirname(wav_output), "generated")
    os.makedirs(generated_dir, exist_ok=True)
    flute_output = os.path.join(generated_dir, f"{base_name}_flute.wav")
    
    print("Generating flute sound from MIDI...")
    flute_path = generator.generate_from_midi(
        midi_path=Path(midi_output),
        output_path=Path(flute_output),
        duration=10  # Adjust duration as needed
    )
    
    if not flute_path:
        print("Error: Flute sound generation failed")
        return
        
    # Step 5: Merge generated flute sound with instrumental
    final_dir = os.path.join(os.path.dirname(wav_output), "final")
    os.makedirs(final_dir, exist_ok=True)
    merged_wav = os.path.join(final_dir, f"{base_name}_final.wav")
    if not merge_audio(instrumental_path, flute_path, merged_wav):
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
    input_file = os.path.join(os.path.dirname(__file__), "data", "input", "song_10.mp3")
    process_audio(input_file)