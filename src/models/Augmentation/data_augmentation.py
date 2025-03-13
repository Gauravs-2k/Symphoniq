#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified data augmentation for both audio and MIDI files for MusicGen training
"""

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
import scipy.signal as signal
import sys

try:
    import pretty_midi
    MIDI_SUPPORT = True
except ImportError:
    print("Warning: pretty_midi not found. MIDI augmentation will be disabled.")
    print("Install with: pip install pretty_midi")
    MIDI_SUPPORT = False

# -----------------------------------------------------------------------------
# Audio Augmentation Functions
# -----------------------------------------------------------------------------

def pitch_shift_audio(audio, sr=32000, n_steps=1.0, normalize=True):
    """Apply pitch shifting to audio"""
    shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    if normalize:
        shifted_audio = librosa.util.normalize(shifted_audio)
    return shifted_audio


def time_stretch_audio(audio, rate=1.0, normalize=True):
    """Apply time stretching to audio"""
    stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
    if normalize:
        stretched_audio = librosa.util.normalize(stretched_audio)
    return stretched_audio


def add_noise_to_audio(audio, noise_level=0.005, noise_type='white', normalize=True):
    """Add noise to audio signal"""
    noisy_audio = audio.copy()
    
    # Generate noise based on type
    if noise_type == 'white':
        noise = np.random.randn(len(audio)) * noise_level
    
    elif noise_type == 'pink':
        white_noise = np.random.randn(len(audio))
        # Using scipy.signal instead of librosa.filters (which doesn't have filtfilt)
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        noise = signal.filtfilt(b, a, white_noise) * noise_level * 2
    
    elif noise_type == 'brown':
        white_noise = np.random.randn(len(audio))
        noise = np.cumsum(white_noise) / np.sqrt(len(audio)) * noise_level
        noise = noise - np.mean(noise)
    
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    # Add noise to audio
    noisy_audio = noisy_audio + noise
    
    # Normalize if requested
    if normalize:
        noisy_audio = librosa.util.normalize(noisy_audio)
    
    return noisy_audio


def crop_audio_segment(audio, sr=32000, segment_duration=5.0, normalize=True):
    """Randomly crop a segment from audio"""
    segment_length = int(segment_duration * sr)
    
    # Check if audio is long enough to crop
    if len(audio) <= segment_length:
        return audio, 0.0, len(audio) / sr
    
    # Choose a random start point
    max_start_idx = len(audio) - segment_length
    start_idx = np.random.randint(0, max_start_idx)
    end_idx = start_idx + segment_length
    
    # Crop the audio
    cropped_audio = audio[start_idx:end_idx].copy()
    
    # Calculate times in seconds
    start_time = start_idx / sr
    end_time = end_idx / sr
    
    # Normalize if requested
    if normalize:
        cropped_audio = librosa.util.normalize(cropped_audio)
        
    return cropped_audio, start_time, end_time


def preprocess_audio_file(audio_path, target_sr=32000, max_length_seconds=10):
    """Load and preprocess audio file"""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Truncate if too long
    max_length = int(max_length_seconds * target_sr)
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    # Normalize
    audio = librosa.util.normalize(audio)
    return audio


# -----------------------------------------------------------------------------
# MIDI Augmentation Functions
# -----------------------------------------------------------------------------

def pitch_shift_midi(midi_data, semitones):
    """Shift the pitch of all notes in a MIDI file"""
    if not MIDI_SUPPORT:
        return None
        
    # Create a new PrettyMIDI object
    new_midi = pretty_midi.PrettyMIDI(resolution=midi_data.resolution, initial_tempo=midi_data.get_tempo_changes()[1][0] if len(midi_data.get_tempo_changes()[1]) > 0 else 120)
    
    # Process each instrument
    for instrument in midi_data.instruments:
        # Create new instrument of the same type
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )
        
        # Shift each note
        for note in instrument.notes:
            # Create a new note with shifted pitch
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=min(max(note.pitch + semitones, 0), 127),  # Ensure pitch stays in MIDI range (0-127)
                start=note.start,
                end=note.end
            )
            new_instrument.notes.append(new_note)
        
        # Copy control changes and pitch bends
        for cc in instrument.control_changes:
            new_instrument.control_changes.append(cc)
        
        for pb in instrument.pitch_bends:
            new_instrument.pitch_bends.append(pb)
        
        # Add the instrument to the new MIDI
        new_midi.instruments.append(new_instrument)
    
    return new_midi


def time_stretch_midi(midi_data, factor):
    """Stretch the timing of a MIDI file"""
    if not MIDI_SUPPORT:
        return None
        
    # Create a new PrettyMIDI object
    new_midi = pretty_midi.PrettyMIDI()
    
    # Process each instrument
    for instrument in midi_data.instruments:
        # Create new instrument of the same type
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )
        
        # Scale timing of each note
        for note in instrument.notes:
            # Scale start and end times
            new_start = note.start / factor
            new_end = note.end / factor
            
            # Create a new note with scaled timing
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)
        
        # Scale timing of control changes and pitch bends
        for cc in instrument.control_changes:
            cc.time /= factor
            new_instrument.control_changes.append(cc)
        
        for pb in instrument.pitch_bends:
            pb.time /= factor
            new_instrument.pitch_bends.append(pb)
        
        # Add the instrument to the new MIDI
        new_midi.instruments.append(new_instrument)
    
    return new_midi


def add_midi_noise(midi_data, timing_noise=0.01, velocity_noise=0.1):
    """Add random variations to MIDI notes (velocity, timing, etc.)"""
    if not MIDI_SUPPORT:
        return None
        
    # Create a new PrettyMIDI object
    new_midi = pretty_midi.PrettyMIDI(resolution=midi_data.resolution, initial_tempo=midi_data.get_tempo_changes()[1][0] if len(midi_data.get_tempo_changes()[1]) > 0 else 120)
    
    # Process each instrument
    for instrument in midi_data.instruments:
        # Create new instrument of the same type
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )
        
        # Add noise to each note
        for note in instrument.notes:
            # Add random variations to timing and velocity
            new_start = note.start + np.random.normal(0, timing_noise)
            new_end = note.end + np.random.normal(0, timing_noise)
            new_velocity = int(note.velocity + np.random.normal(0, velocity_noise * 127))
            
            # Ensure values stay in valid ranges
            new_start = max(0, new_start)
            new_end = max(new_start + 0.01, new_end)  # Ensure note has positive duration
            new_velocity = min(max(new_velocity, 1), 127)  # MIDI velocity: 1-127
            
            # Create a new note with noisy parameters
            new_note = pretty_midi.Note(
                velocity=new_velocity,
                pitch=note.pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)
        
        # Copy control changes and pitch bends (without noise)
        for cc in instrument.control_changes:
            new_instrument.control_changes.append(cc)
        
        for pb in instrument.pitch_bends:
            new_instrument.pitch_bends.append(pb)
        
        # Add the instrument to the new MIDI
        new_midi.instruments.append(new_instrument)
    
    return new_midi


def crop_midi_segment(midi_data, start_time, end_time):
    """Extract a segment of a MIDI file"""
    if not MIDI_SUPPORT:
        return None
        
    # Create a new PrettyMIDI object
    new_midi = pretty_midi.PrettyMIDI()
    
    # Process each instrument
    for instrument in midi_data.instruments:
        # Create new instrument of the same type
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )
        
        # Filter notes that overlap with the segment
        for note in instrument.notes:
            # Check if the note overlaps with the segment
            if note.end > start_time and note.start < end_time:
                # Adjust start and end times relative to the segment
                new_start = max(note.start - start_time, 0)
                new_end = min(note.end - start_time, end_time - start_time)
                
                # Create a new note
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=new_start,
                    end=new_end
                )
                new_instrument.notes.append(new_note)
        
        # Filter control changes and pitch bends in the segment
        for cc in instrument.control_changes:
            if start_time <= cc.time < end_time:
                new_cc = pretty_midi.ControlChange(
                    number=cc.number,
                    value=cc.value,
                    time=cc.time - start_time
                )
                new_instrument.control_changes.append(new_cc)
        
        for pb in instrument.pitch_bends:
            if start_time <= pb.time < end_time:
                new_pb = pretty_midi.PitchBend(
                    pitch=pb.pitch,
                    time=pb.time - start_time
                )
                new_instrument.pitch_bends.append(new_pb)
        
        # Add the instrument to the new MIDI if it has notes
        if new_instrument.notes:
            new_midi.instruments.append(new_instrument)
    
    return new_midi


# -----------------------------------------------------------------------------
# Main Augmentation Functions
# -----------------------------------------------------------------------------

def create_instrument_prompt(piece_name, instrument="flute", extra=""):
    """Create a detailed prompt for instrument generation"""
    base_prompt = (
        f"Generate a solo {instrument} performance playing {piece_name}. "
        f"The {instrument} should play with proper breath control, fluid phrasing, "
        f"clear articulation, and accurate timing."
    )
    
    if extra:
        base_prompt += f" {extra}"
        
    base_prompt += (
        f" This should sound like a professional {instrument} recording "
        f"with natural dynamics and expressive interpretation."
    )
    
    return base_prompt


def augment_audio_files(audio_dir, output_dir, instrument="flute", 
                        pitch_shifts=None, time_stretches=None, 
                        noise_params=None, crop_params=None,
                        max_examples=None):
    """Process and augment audio files"""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir) / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all WAV files
    audio_files = list(audio_dir.glob("*.wav"))
    if max_examples:
        audio_files = audio_files[:max_examples]
        
    print(f"Processing {len(audio_files)} audio files...")
    
    # Process original files first and save them
    for audio_path in tqdm(audio_files, desc="Processing original audio files"):
        try:
            # Load and process audio
            audio = preprocess_audio_file(audio_path)
            
            # Save original
            output_path = output_dir / f"{audio_path.stem}_original.wav"
            scipy.io.wavfile.write(
                output_path,
                rate=32000,
                data=(audio * 32767).astype(np.int16)
            )
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
    
    # Apply pitch shifting
    if pitch_shifts:
        for semitones in pitch_shifts:
            for audio_path in tqdm(audio_files, desc=f"Pitch shifting: {semitones} semitones"):
                try:
                    # Load audio
                    audio = preprocess_audio_file(audio_path)
                    
                    # Apply pitch shifting
                    shifted_audio = pitch_shift_audio(audio, n_steps=semitones)
                    
                    # Create output filename
                    shift_str = f"up_{semitones}" if semitones > 0 else f"down_{abs(semitones)}"
                    output_path = output_dir / f"{audio_path.stem}_pitch_{shift_str}.wav"
                    
                    # Save as WAV
                    scipy.io.wavfile.write(
                        output_path,
                        rate=32000,
                        data=(shifted_audio * 32767).astype(np.int16)
                    )
                    
                except Exception as e:
                    print(f"Error pitch shifting {audio_path}: {str(e)}")
    
    # Apply time stretching
    if time_stretches:
        for factor in time_stretches:
            for audio_path in tqdm(audio_files, desc=f"Time stretching: {factor}x"):
                try:
                    # Load audio
                    audio = preprocess_audio_file(audio_path)
                    
                    # Apply time stretching
                    stretched_audio = time_stretch_audio(audio, rate=factor)
                    
                    # Create output filename
                    stretch_str = f"faster_{factor}" if factor > 1 else f"slower_{1/factor:.2f}"
                    output_path = output_dir / f"{audio_path.stem}_tempo_{stretch_str}.wav"
                    
                    # Save as WAV
                    scipy.io.wavfile.write(
                        output_path,
                        rate=32000,
                        data=(stretched_audio * 32767).astype(np.int16)
                    )
                    
                except Exception as e:
                    print(f"Error time stretching {audio_path}: {str(e)}")
    
    # Apply noise addition
    if noise_params:
        noise_types = noise_params.get('types', ['white', 'pink'])
        noise_levels = noise_params.get('levels', [0.002, 0.005])
        
        for noise_type in noise_types:
            for level in noise_levels:
                for audio_path in tqdm(audio_files, desc=f"{noise_type} noise (level={level:.4f})"):
                    try:
                        # Load audio
                        audio = preprocess_audio_file(audio_path)
                        
                        # Apply noise
                        noisy_audio = add_noise_to_audio(audio, 
                                                        noise_level=level, 
                                                        noise_type=noise_type)
                        
                        # Create output filename
                        output_path = output_dir / f"{audio_path.stem}_noise_{noise_type}_{level:.4f}.wav"
                        
                        # Save as WAV
                        scipy.io.wavfile.write(
                            output_path,
                            rate=32000,
                            data=(noisy_audio * 32767).astype(np.int16)
                        )
                        
                    except Exception as e:
                        print(f"Error adding noise to {audio_path}: {str(e)}")
    
    # Apply random cropping
    if crop_params:
        num_crops = crop_params.get('num_crops', 3)
        durations = crop_params.get('durations', [3.0, 5.0, 7.0])
        
        for audio_path in tqdm(audio_files, desc="Random cropping"):
            try:
                # Load audio
                audio = preprocess_audio_file(audio_path, max_length_seconds=30)  # Allow longer for cropping
                
                # Create multiple crops
                for i in range(num_crops):
                    # Randomly select segment duration
                    segment_duration = np.random.choice(durations)
                    
                    # Crop the audio
                    cropped, start_time, end_time = crop_audio_segment(
                        audio, 
                        segment_duration=segment_duration
                    )
                    
                    # Create output filename
                    output_path = output_dir / f"{audio_path.stem}_crop_{i+1}_{start_time:.1f}s_to_{end_time:.1f}s.wav"
                    
                    # Save as WAV
                    scipy.io.wavfile.write(
                        output_path,
                        rate=32000,
                        data=(cropped * 32767).astype(np.int16)
                    )
                    
            except Exception as e:
                print(f"Error cropping {audio_path}: {str(e)}")
    
    print(f"Audio augmentation complete. Files saved to {output_dir}")


def augment_midi_files(midi_dir, output_dir,
                       pitch_shifts=None, time_stretches=None, 
                       add_noise=False, random_crops=0,
                       max_examples=None):
    """Process and augment MIDI files"""
    if not MIDI_SUPPORT:
        print("MIDI augmentation skipped: pretty_midi module not available")
        return
        
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir) / "midi"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all MIDI files
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    if max_examples:
        midi_files = midi_files[:max_examples]
        
    print(f"Processing {len(midi_files)} MIDI files...")
    
    # Process each MIDI file
    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Save original copy
            output_path = output_dir / f"{midi_path.stem}_original.mid"
            midi_data.write(str(output_path))
            
            # 1. Pitch shifting
            if pitch_shifts:
                for semitones in pitch_shifts:
                    try:
                        shifted_midi = pitch_shift_midi(midi_data, semitones)
                        shift_str = f"up_{semitones}" if semitones > 0 else f"down_{abs(semitones)}"
                        output_path = output_dir / f"{midi_path.stem}_pitch_{shift_str}.mid"
                        shifted_midi.write(str(output_path))
                    except Exception as e:
                        print(f"Error in pitch shifting {midi_path}: {e}")
            
            # 2. Time stretching
            if time_stretches:
                for factor in time_stretches:
                    try:
                        stretched_midi = time_stretch_midi(midi_data, factor)
                        stretch_str = f"faster_{factor}" if factor > 1 else f"slower_{1/factor:.2f}"
                        output_path = output_dir / f"{midi_path.stem}_tempo_{stretch_str}.mid"
                        stretched_midi.write(str(output_path))
                    except Exception as e:
                        print(f"Error in time stretching {midi_path}: {e}")
            
            # 3. Add noise variation
            if add_noise:
                try:
                    # Light noise
                    noisy_midi_light = add_midi_noise(midi_data, timing_noise=0.01, velocity_noise=0.05)
                    output_path = output_dir / f"{midi_path.stem}_noise_light.mid"
                    noisy_midi_light.write(str(output_path))
                    
                    # Medium noise
                    noisy_midi_med = add_midi_noise(midi_data, timing_noise=0.02, velocity_noise=0.1)
                    output_path = output_dir / f"{midi_path.stem}_noise_medium.mid"
                    noisy_midi_med.write(str(output_path))
                except Exception as e:
                    print(f"Error adding noise to {midi_path}: {e}")
            
            # 4. Random cropping
            if random_crops > 0:
                try:
                    duration = midi_data.get_end_time()
                    if duration > 5.0:  # Only crop if long enough
                        for i in range(random_crops):
                            # Select a random segment (3-10 seconds long)
                            segment_length = min(duration, np.random.uniform(3.0, 10.0))
                            max_start = max(0, duration - segment_length)
                            start_time = np.random.uniform(0, max_start)
                            end_time = start_time + segment_length
                            
                            # Crop the segment
                            cropped_midi = crop_midi_segment(midi_data, start_time, end_time)
                            output_path = output_dir / f"{midi_path.stem}_crop_{i+1}_{start_time:.1f}s_to_{end_time:.1f}s.mid"
                            cropped_midi.write(str(output_path))
                except Exception as e:
                    print(f"Error in cropping {midi_path}: {e}")
            
        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
    
    print(f"MIDI augmentation complete. Files saved to {output_dir}")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import scipy.io.wavfile
    
    parser = argparse.ArgumentParser(description='Create augmented audio and MIDI datasets')
    parser.add_argument('--input_dir', type=str, default="/Users/gauravs/Documents/Symphoniq/src/data/training",
                       help='Directory containing audio and MIDI files')
    parser.add_argument('--output_dir', type=str, default="/Users/gauravs/Documents/Symphoniq/src/data/augmented",
                       help='Directory to save augmented examples')
    parser.add_argument('--instrument', type=str, default="flute",
                       help='Instrument name for prompts')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to process')
    
    # Audio subdirectory
    parser.add_argument('--audio_subdir', type=str, default="audio",
                       help='Subdirectory in input_dir containing audio files')
    # MIDI subdirectory
    parser.add_argument('--midi_subdir', type=str, default="midi",
                       help='Subdirectory in input_dir containing MIDI files')
    
    # Augmentation control
    parser.add_argument('--no_pitch_shift', action='store_true',
                       help='Disable pitch shifting')
    parser.add_argument('--no_time_stretch', action='store_true',
                       help='Disable time stretching')
    parser.add_argument('--no_noise', action='store_true',
                       help='Disable noise addition')
    parser.add_argument('--no_crop', action='store_true',
                       help='Disable random cropping')
    parser.add_argument('--num_crops', type=int, default=3,
                       help='Number of crops per file')
    
    # File type selection
    parser.add_argument('--audio_only', action='store_true',
                       help='Process only audio files')
    parser.add_argument('--midi_only', action='store_true',
                       help='Process only MIDI files')
    
    args = parser.parse_args()
    
    # Configure augmentations based on arguments
    pitch_shifts = None if args.no_pitch_shift else [-2, -1, 1, 2]
    time_stretches = None if args.no_time_stretch else [0.8, 0.9, 1.1, 1.2]
    noise_params = None if args.no_noise else {
        'types': ['white', 'pink'], 
        'levels': [0.002, 0.005]
    }
    crop_params = None if args.no_crop else {
        'num_crops': args.num_crops,
        'durations': [3.0, 5.0, 7.0]
    }
    random_crops = 0 if args.no_crop else args.num_crops
    
    # Print configuration
    print("\nData Augmentation Configuration:")
    print("---------------------------------")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Instrument: {args.instrument}")
    
    # Print augmentation settings without nested f-strings that can cause errors
    print("\nAugmentation settings:")
    print(f"- Pitch shifting: {'Disabled' if pitch_shifts is None else str(pitch_shifts)}")
    print(f"- Time stretching: {'Disabled' if time_stretches is None else str(time_stretches)}")
    
    # Handle noise params print safely
    noise_info = "Disabled" if noise_params is None else f"{noise_params.get('types')} at levels {noise_params.get('levels')}"
    print(f"- Noise addition: {noise_info}")
    
    # Handle crop params print safely
    crop_info = "Disabled" if crop_params is None else f"{crop_params.get('num_crops')} crops per file, durations: {crop_params.get('durations')}s"
    print(f"- Random cropping: {crop_info}")
    
    # Process audio files
    if not args.midi_only:
        audio_dir = Path(args.input_dir) / args.audio_subdir
        if audio_dir.exists():
            print("\nProcessing audio files...")
            augment_audio_files(
                audio_dir=audio_dir,
                output_dir=args.output_dir,
                instrument=args.instrument,
                pitch_shifts=pitch_shifts,
                time_stretches=time_stretches,
                noise_params=noise_params,
                crop_params=crop_params,
                max_examples=args.max_examples
            )
        else:
            print(f"\nAudio directory not found: {audio_dir}")
    
    # Process MIDI files
    if not args.audio_only and MIDI_SUPPORT:
        midi_dir = Path(args.input_dir) / args.midi_subdir
        if midi_dir.exists():
            print("\nProcessing MIDI files...")
            augment_midi_files(
                midi_dir=midi_dir,
                output_dir=args.output_dir,
                pitch_shifts=pitch_shifts,
                time_stretches=time_stretches,
                add_noise=not args.no_noise,
                random_crops=random_crops,
                max_examples=args.max_examples
            )
        else:
            print(f"\nMIDI directory not found: {midi_dir}")
    
    print("\nAugmentation complete!")