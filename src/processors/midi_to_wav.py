import os
from midi2audio import FluidSynth

def midi_to_wav(midi_file, output_wav):
    """
    Convert MIDI file to WAV using FluidSynth
    Args:
        midi_file (str): Path to input MIDI file
        output_wav (str): Path to output WAV file
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize FluidSynth with default soundfont
        fs = FluidSynth()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_wav), exist_ok=True)
        
        # Convert MIDI to WAV
        fs.midi_to_audio(midi_file, output_wav)
        print(f"Successfully converted MIDI to WAV: {output_wav}")
        return True
        
    except Exception as e:
        print(f"Error converting MIDI to WAV: {str(e)}")
        return False