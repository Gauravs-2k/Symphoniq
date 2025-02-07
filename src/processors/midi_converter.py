import os
import librosa
import pretty_midi
import numpy as np

def vocal_to_midi(input_file, output_file, min_note='C2', max_note='C7'):
    try:
        print(f"Processing {input_file}...")
        
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load and process audio with specific sample rate
        y, sr = librosa.load(input_file, sr=22050)  # Fixed sample rate
        
        # Extract pitch using safer parameters
        print("Extracting pitch...")
        pitches = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz(min_note),
            fmax=librosa.note_to_hz(max_note),
            sr=sr
        )[0]  # Only take first return value
        
        # Filter out None values and convert to MIDI
        print("Converting to MIDI...")
        valid_pitches = pitches[~np.isnan(pitches)] if pitches is not None else np.array([])
        midi_numbers = librosa.hz_to_midi(valid_pitches)
        
        if len(midi_numbers) == 0:
            raise ValueError("No valid pitches detected in audio")
            
        # Create MIDI file
        midi = pretty_midi.PrettyMIDI()
        
        # Create instrument (flute)
        flute_program = pretty_midi.instrument_name_to_program('Flute')
        instrument = pretty_midi.Instrument(program=flute_program)
        
        # Add notes with more robust timing
        time = 0
        duration = 0.25  # Quarter note duration
        
        for pitch in midi_numbers:
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(round(pitch)),
                start=time,
                end=time + duration
            )
            instrument.notes.append(note)
            time += duration
            
        # Add instrument to MIDI
        midi.instruments.append(instrument)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save MIDI file
        midi.write(output_file)
        print(f"MIDI file saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    input_file = "../data/seperated/eterna-cancao-wav-12569_vocals.wav"
    output_file = "midi/vocal_melody.mid"
    vocal_to_midi(input_file, output_file)