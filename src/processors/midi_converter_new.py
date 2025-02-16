import os
import librosa
import numpy as np
from mido import Message, MidiFile, MidiTrack

def vocal_to_midi(input_file, output_file, min_note='C2', max_note='C7'):
    try:
        print(f"Processing {input_file}...")
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Load the WAV file
        try:
            y, sr = librosa.load(input_file, sr=None)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False

        # Extract onset frames (note timing)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')

        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Create a new MIDI file
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Set instrument (Program Change Message - Piano)
        track.append(Message('program_change', program=0, time=0))

        # Iterate through onset times
        prev_time = 0
        for onset in onset_frames:
            frame = librosa.time_to_frames(onset, sr=sr)
            if frame < magnitudes.shape[1]:  # Add bounds checking
                pitch_idx = np.argmax(magnitudes[:, frame])
                if pitch_idx < pitches.shape[0]:  # Add bounds checking
                    pitch = pitches[pitch_idx, frame]

                    if pitch > 0:
                        midi_note = int(librosa.hz_to_midi(pitch))
                        midi_time = int(onset * 480)  # Convert onset time to MIDI ticks

                        # Add note_on and note_off events
                        track.append(Message('note_on', note=midi_note, velocity=64, time=midi_time - prev_time))
                        track.append(Message('note_off', note=midi_note, velocity=64, time=240))  # Quarter note duration

                        prev_time = midi_time

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save MIDI file
        mid.save(output_file)
        print(f"MIDI file saved: {output_file}")
        return True

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = os.path.join(os.getcwd(), "separated", "eterna-cancao-wav-12569_vocals.wav")
    output_file = os.path.join(os.getcwd(), "midi", "vocal_melody.mid")
    vocal_to_midi(input_file, output_file)