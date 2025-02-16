from pathlib import Path
import sys
import pretty_midi

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.instrumental_generator import InstrumentalGenerator
from src.processors.validate_midi import validate_midi_file

def test_midi_generation():
    generator = InstrumentalGenerator()
    
    midi_path = Path("../data/midi/vocal_melody.mid")
    
    # Validate MIDI file
    is_valid, error_msg = validate_midi_file(midi_path)
    if not is_valid:
        print(f"Invalid MIDI file: {error_msg}")
        return
    
    # Load MIDI to get duration
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    total_duration = int(midi_data.get_end_time())
    
    # Process in chunks of 30 seconds
    chunk_duration = 30
    num_chunks = (total_duration + chunk_duration - 1) // chunk_duration
    
    print(f"Total duration: {total_duration} seconds")
    print(f"Processing in {num_chunks} chunks...")
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_output = Path(f"../data/processed/instruments/midi_generated_part{i+1}.wav")
        
        print(f"\nGenerating chunk {i+1}/{num_chunks} ({start_time} - {start_time + chunk_duration} seconds)")
        result = generator.generate_from_midi(
            midi_path,
            chunk_output,
            start_time=start_time,
            duration=chunk_duration
        )
        
        if result:
            print(f"Generated chunk {i+1} saved to: {result}")
        else:
            print(f"Failed to generate chunk {i+1}")

if __name__ == "__main__":
    test_midi_generation()