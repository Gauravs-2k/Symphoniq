from converters.mp3_to_wav import convert_to_wav
from converters.wav_to_mp3 import convert_to_mp3
from processors.vocal_seperator import separate_audio
from processors.midi_converter_new import vocal_to_midi
from processors.midi_to_wav import midi_to_wav
from processors.audio_merger import merge_audio
import os

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
        
    # Step 4: Convert MIDI back to WAV
    midi_wav = os.path.join(midi_dir, f"{base_name}_midi_audio.wav")
    if not midi_to_wav(midi_output, midi_wav):
        print("Error: MIDI to WAV conversion failed")
        return
        
    # Step 5: Merge MIDI-generated WAV with instrumental
    final_dir = os.path.join(os.path.dirname(wav_output), "final")
    os.makedirs(final_dir, exist_ok=True)
    merged_wav = os.path.join(final_dir, f"{base_name}_final.wav")
    if not merge_audio(instrumental_path, midi_wav, merged_wav):
        print("Error: Audio merging failed")
        return
        
    print(f"Processing complete! Final MP3: {final_dir}")
    return final_dir

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), "data", "input", "eterna-cancao-wav-12569.mp3")
    process_audio(input_file)