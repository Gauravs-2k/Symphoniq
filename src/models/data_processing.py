#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing utilities for MusicGen fine-tuning
"""

import os
import numpy as np
import torch
import librosa
import pretty_midi
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
import soundfile as sf # Add this import to check audio properties

def validate_audio_files(audio_dir):
    """
    Validate audio files in the directory and print diagnostics
    
    Args:
        audio_dir: Directory containing audio files (.wav)
        
    Returns:
        Dictionary with validation results
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav"))
    
    print(f"Validating {len(audio_files)} audio files...")
    
    results = {
        "total": len(audio_files),
        "valid": 0,
        "sample_rates": {},
        "channels": {},
        "durations": [],
        "issues": []
    }
    
    for audio_path in tqdm(audio_files):
        try:
            # Get audio metadata without loading entire file
            info = sf.info(audio_path)
            
            # Count sample rates
            sr = info.samplerate
            results["sample_rates"][sr] = results["sample_rates"].get(sr, 0) + 1
            
            # Count channels 
            ch = info.channels
            results["channels"][ch] = results["channels"].get(ch, 0) + 1
            
            # Track durations
            duration = info.duration
            results["durations"].append(duration)
            
            # Check for issues
            issues = []
            if sr != 32000 and sr != 44100:
                issues.append(f"Non-standard sample rate: {sr}Hz")
            if ch > 1:
                issues.append(f"Multi-channel audio: {ch} channels")
            if duration < 1.0:
                issues.append(f"Very short audio: {duration:.2f}s")
            elif duration > 30.0:
                issues.append(f"Very long audio: {duration:.2f}s")
                
            if issues:
                results["issues"].append((str(audio_path), issues))
            else:
                results["valid"] += 1
                
        except Exception as e:
            results["issues"].append((str(audio_path), [f"Error: {str(e)}"]))
    
    # Print summary
    print(f"\nAudio validation summary:")
    print(f"- Total files: {results['total']}")
    print(f"- Valid files: {results['valid']}")
    print(f"- Sample rates: {dict(sorted(results['sample_rates'].items()))}")
    print(f"- Channel counts: {dict(sorted(results['channels'].items()))}")
    
    if results["durations"]:
        min_dur = min(results["durations"])
        max_dur = max(results["durations"])
        avg_dur = sum(results["durations"]) / len(results["durations"])
        print(f"- Duration range: {min_dur:.2f}s - {max_dur:.2f}s (avg: {avg_dur:.2f}s)")
    
    if results["issues"]:
        print("\nIssues found:")
        for path, issues in results["issues"][:10]:  # Show first 10 issues
            print(f"- {Path(path).name}: {', '.join(issues)}")
        if len(results["issues"]) > 10:
            print(f"  ... and {len(results["issues"])-10} more files with issues")
    
    return results

# Update preprocess_audio_file to add resampling diagnostics
def preprocess_audio_file(audio_path, target_sr=32000, max_length_seconds=10):
    """
    Load and preprocess an audio file for MusicGen training
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (32kHz for MusicGen)
        max_length_seconds: Maximum audio length in seconds
        
    Returns:
        Preprocessed audio as numpy array
    """
    # Check original file properties first
    info = sf.info(audio_path)
    original_sr = info.samplerate
    
    # Load audio with resampling
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    if original_sr != target_sr:
        print(f"Resampled {audio_path.name} from {original_sr}Hz to {target_sr}Hz")
    
    # Trim to specified length
    original_length = len(audio)
    max_length = max_length_seconds * target_sr
    if len(audio) > max_length:
        audio = audio[:max_length]
        print(f"Trimmed {audio_path.name} from {original_length/target_sr:.2f}s to {max_length/target_sr:.2f}s")
    
    # Normalize audio to [-1, 1]
    original_range = (audio.min(), audio.max())
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = librosa.util.normalize(audio)
        print(f"Normalized {audio_path.name} from range {original_range} to [-1,1]")
    
    return audio

def preprocess_audio_file(audio_path, target_sr=32000, max_length_seconds=10):
    """
    Load and preprocess an audio file for MusicGen training
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (32kHz for MusicGen)
        max_length_seconds: Maximum audio length in seconds
        
    Returns:
        Preprocessed audio as numpy array
    """
    # Load audio with resampling
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Trim to specified length
    max_length = max_length_seconds * target_sr
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    # Normalize audio to [-1, 1]
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = librosa.util.normalize(audio)
    
    return audio

def extract_notes_from_midi(midi_path, max_notes=40):
    """
    Extract notes from a MIDI file
    
    Args:
        midi_path: Path to MIDI file
        max_notes: Maximum number of notes to extract
        
    Returns:
        List of note names
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        notes = []
        
        for instr in midi_data.instruments:
            if not instr.is_drum:
                for note in instr.notes[:max_notes]:
                    pitch_name = pretty_midi.note_number_to_name(note.pitch)
                    notes.append(pitch_name)
                    
        return notes
    except Exception as e:
        print(f"Error processing MIDI {midi_path}: {str(e)}")
        return []

def build_audio_dataset(audio_dir, instrument="flute", max_examples=None):
    """
    Build dataset from audio files
    
    Args:
        audio_dir: Directory containing audio files (.wav)
        instrument: Instrument name for prompts
        max_examples: Maximum number of examples to include
        
    Returns:
        Dataset object with processed examples
    """
    audio_dir = Path(audio_dir)
    dataset_entries = []
    
    # Get all WAV files
    audio_files = list(audio_dir.glob("*.wav"))
    if max_examples:
        audio_files = audio_files[:max_examples]
    
    print(f"Processing {len(audio_files)} audio files for {instrument}...")
    
    for audio_path in tqdm(audio_files):
        try:
            # Create prompt from filename
            piece_name = audio_path.stem.replace('_', ' ')
            prompt = create_instrument_prompt(piece_name, instrument)
            
            # Load and process audio
            audio = preprocess_audio_file(audio_path)
            
            # Add to dataset
            dataset_entries.append({
                "prompt": prompt,
                "audio": audio.tolist(),  # Convert to list for serialization
                "audio_path": str(audio_path)
            })
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
    
    return Dataset.from_list(dataset_entries)

def create_instrument_prompt(piece_name, instrument="flute"):
    """Create a detailed prompt for instrument generation"""
    return (
        f"Generate a solo {instrument} performance playing {piece_name}. "
        f"The {instrument} should play with proper breath control, fluid phrasing, "
        f"clear articulation, and accurate timing. This should sound like a "
        f"professional {instrument} recording with natural dynamics and expressive interpretation."
    )

def encode_audio_for_musicgen(audio_tensor, audio_encoder, device="cpu"):
    """
    Encode audio using MusicGen's audio encoder
    
    Args:
        audio_tensor: Audio tensor [batch, seq_len]
        audio_encoder: MusicGen audio encoder
        device: Device to run encoding on
        
    Returns:
        Properly formatted audio codes for training
    """
    with torch.no_grad():
        # Shape: [batch, seq_len]
        if len(audio_tensor.shape) == 1:
            # Add batch dimension if missing
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Get audio codes
        audio_codes = audio_encoder(audio_tensor.to(device)).audio_codes
        
        # Format the audio codes properly based on shape
        if len(audio_codes.shape) == 4:  # [1, 1, codebooks, seq_len]
            # Reshape to [seq_len] by flattening the representation in the expected way
            # We need to interleave the codebooks to match the expected format
            seq_len = audio_codes.shape[-1]
            # Just use the first codebook as labels (this is the common approach for training)
            audio_tokens = audio_codes[0, 0, 0, :].view(-1)
        else:  # [batch, codebooks, seq_len] 
            # Just use the first codebook as labels
            audio_tokens = audio_codes[0, 0, :].view(-1)
            
    return audio_tokens

def process_example_for_training(example, processor, audio_encoder, device="cpu"):
    """
    Process a single example for MusicGen training
    
    Args:
        example: Dict with 'prompt' and 'audio' keys
        processor: MusicGen processor
        audio_encoder: MusicGen audio encoder
        device: Device to run processing on
        
    Returns:
        Dict with processed tensors for training
    """
    # Process text prompt
    text_inputs = processor.tokenizer(
        example["prompt"],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    # Process audio
    audio_array = np.array(example["audio"])
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    
    # Get audio features
    audio_inputs = processor.feature_extractor(
        audio_tensor, 
        sampling_rate=32000,
        return_tensors="pt"
    )
    
    # Get encoded audio tokens
    audio_tokens = encode_audio_for_musicgen(
        audio_inputs.input_values,
        audio_encoder,
        device=device
    )
    
    # Return processed tensors
    return {
        "input_ids": text_inputs.input_ids.squeeze(0),
        "attention_mask": text_inputs.attention_mask.squeeze(0),
        "labels": audio_tokens
    }