#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare original and fine-tuned MusicGen models using appropriate audio metrics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.spatial.distance import cosine
from scipy.io import wavfile
import pandas as pd
import seaborn as sns

# Set paths to models
original_model_path = "/Users/gauravs/Documents/Symphoniq/src/models/cached_models"
finetuned_model_path = "/Users/gauravs/Documents/Symphoniq/src/models/midi_to_flute_model/final_model"

# Set path for test data
midi_dir = "/Users/gauravs/Documents/Symphoniq/src/data/testing/midi"
reference_audio_dir = "/Users/gauravs/Documents/Symphoniq/src/data/testing/audio"
output_dir = Path("/Users/gauravs/Documents/Symphoniq/src/models/comparisons")
output_dir.mkdir(exist_ok=True)

def load_models_and_processor(original_path, finetuned_path):
    """Load both models and the processor"""
    # Load the processor from either path
    processor = AutoProcessor.from_pretrained(original_path)
    
    # Load models
    original_model = MusicgenForConditionalGeneration.from_pretrained(original_path)
    finetuned_model = MusicgenForConditionalGeneration.from_pretrained(finetuned_path)
    
    # Set models to evaluation mode
    original_model.eval()
    finetuned_model.eval()
    
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model.to(device)
    finetuned_model.to(device)
    
    return original_model, finetuned_model, processor, device

def process_midi_to_prompt(midi_path):
    """Convert MIDI file to text prompt for model input"""
    # This should match the processing in your training code
    from pretty_midi import PrettyMIDI
    import librosa
    
    try:
        # Parse MIDI - convert Path to string to avoid the PosixPath read error
        midi_data = PrettyMIDI(str(midi_path))
        
        # Extract key information from MIDI
        total_time = int(midi_data.get_end_time())
        
        # Get instruments
        instruments = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                inst_name = instrument.name if instrument.name else "Piano"
                instruments.append(inst_name)
        
        # Count notes per instrument
        num_notes = 0
        pitch_range = [128, 0]  # [min, max]
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                num_notes += len(instrument.notes)
                for note in instrument.notes:
                    pitch_range[0] = min(pitch_range[0], note.pitch)
                    pitch_range[1] = max(pitch_range[1], note.pitch)
        
        # Get time signature
        time_sig = "4/4"  # Default
        for ts in midi_data.time_signature_changes:
            time_sig = f"{ts.numerator}/{ts.denominator}"
            break  # Just use the first one
        
        # Get tempo safely
        tempo = 120  # Default
        try:
            _, tempos = midi_data.get_tempo_changes()
            if len(tempos) > 0:
                tempo = int(tempos[0])
        except (AttributeError, ValueError):
            pass
        
        # Create text prompt
        instrument_text = ", ".join(instruments[:3]) if instruments else "Piano"
        prompt = (
            f"Create a flute performance based on this MIDI. "
            f"The original is for {instrument_text}, with {num_notes} notes "
            f"ranging from {librosa.midi_to_note(pitch_range[0])} to "
            f"{librosa.midi_to_note(pitch_range[1])}, in {time_sig} at {tempo} BPM, "
            f"lasting {total_time} seconds. "
            f"The flute should play expressively with clear articulation and good breath control."
        )
        
        return prompt
    except Exception as e:
        print(f"Error processing MIDI {midi_path}: {e}")
        return "Create a flute performance based on a MIDI file."

def compute_spectral_metrics(audio1, audio2, sr=32000):
    """Compute spectral metrics between two audio samples"""
    # Compute MFCCs
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=20)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=20)
    
    # Compute mean MFCCs over time
    mfcc1_mean = np.mean(mfcc1, axis=1)
    mfcc2_mean = np.mean(mfcc2, axis=1)
    
    # Compute cosine similarity between MFCCs
    mfcc_similarity = 1 - cosine(mfcc1_mean, mfcc2_mean)
    
    # Compute spectrograms
    S1 = np.abs(librosa.stft(audio1))
    S2 = np.abs(librosa.stft(audio2))
    
    # Compute spectral contrast
    contrast1 = np.mean(librosa.feature.spectral_contrast(S=S1, sr=sr))
    contrast2 = np.mean(librosa.feature.spectral_contrast(S=S2, sr=sr))
    contrast_diff = abs(contrast1 - contrast2)
    
    # Compute spectral flatness
    flatness1 = np.mean(librosa.feature.spectral_flatness(S=S1))
    flatness2 = np.mean(librosa.feature.spectral_flatness(S=S2))
    flatness_diff = abs(flatness1 - flatness2)
    
    return {
        "mfcc_similarity": mfcc_similarity,
        "spectral_contrast_diff": contrast_diff,
        "spectral_flatness_diff": flatness_diff
    }

def generate_audio_and_compare(processor, original_model, finetuned_model, midi_paths, device, sample_count=10):
    """Generate audio from both models and compare metrics"""
    results = []
    
    for i, midi_path in enumerate(tqdm(midi_paths[:sample_count])):
        midi_path = Path(midi_path)
        prompt = process_midi_to_prompt(midi_path)
        
        # Process prompt for model input
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        
        # Generate audio with both models
        with torch.no_grad():
            # Generate with original model
            original_outputs = original_model.generate(
                **inputs, 
                max_new_tokens=500,
                do_sample=True,
                guidance_scale=3.0
            )
            
            # Generate with fine-tuned model
            finetuned_outputs = finetuned_model.generate(
                **inputs, 
                max_new_tokens=500,
                do_sample=True,
                guidance_scale=3.0
            )
        
        # Process the output to get audio waveforms
        try:
            # Extract audio from outputs based on model version
            if hasattr(original_outputs, "audio_values"):
                original_audio = original_outputs.audio_values.cpu().numpy()[0]
                finetuned_audio = finetuned_outputs.audio_values.cpu().numpy()[0]
            else:
                original_audio = original_outputs.cpu().numpy()[0]
                finetuned_audio = finetuned_outputs.cpu().numpy()[0]
            
            # Debug shapes
            print(f"Original audio shape before processing: {original_audio.shape}")
            
            # Fix dimensions - ensure we have a 1D array
            if len(original_audio.shape) > 1:
                if original_audio.shape[0] == 1:  # If first dimension is 1, squeeze it
                    original_audio = original_audio.squeeze(0)
                    finetuned_audio = finetuned_audio.squeeze(0)
                elif original_audio.shape[1] == 1:  # If second dimension is 1, squeeze it
                    original_audio = original_audio.squeeze(1)
                    finetuned_audio = finetuned_audio.squeeze(1)
            
            # Debug information
            print(f"Original audio shape after processing: {original_audio.shape}, dtype: {original_audio.dtype}")
            print(f"Audio range: min={np.min(original_audio)}, max={np.max(original_audio)}")
            
            # Normalize audio to [-1, 1] range
            original_audio = np.clip(original_audio / max(np.max(np.abs(original_audio)), 1e-10), -0.99, 0.99)
            finetuned_audio = np.clip(finetuned_audio / max(np.max(np.abs(finetuned_audio)), 1e-10), -0.99, 0.99)
            
            # For scipy.io.wavfile, scale to int16 range safely
            original_audio_int = (original_audio * 32767).astype(np.int16)
            finetuned_audio_int = (finetuned_audio * 32767).astype(np.int16)
            
            # Save generated audio for comparison
            sample_name = midi_path.stem
            original_output_path = output_dir / f"{sample_name}_original.wav"
            finetuned_output_path = output_dir / f"{sample_name}_finetuned.wav"
            
            # Use scipy to write WAV files
            wavfile.write(str(original_output_path), 32000, original_audio_int)
            wavfile.write(str(finetuned_output_path), 32000, finetuned_audio_int)
            
            print(f"Successfully saved audio files to {original_output_path}")
            
            # Rest of the function (metrics computation) remains the same but use the floating-point
            # audio for spectral metric calculations
            
            # Try to find reference audio if available
            reference_path = Path(reference_audio_dir) / f"{sample_name}.wav"
            if reference_path.exists():
                reference_audio, sr = librosa.load(str(reference_path), sr=32000)
                
                # Compute metrics comparing to reference
                original_ref_metrics = compute_spectral_metrics(original_audio, reference_audio)
                finetuned_ref_metrics = compute_spectral_metrics(finetuned_audio, reference_audio)
                
                # Add reference comparison
                results.append({
                    'sample': sample_name,
                    'model': 'Original',
                    'reference_mfcc_similarity': original_ref_metrics['mfcc_similarity'],
                    'reference_spectral_contrast': original_ref_metrics['spectral_contrast_diff'],
                    'reference_spectral_flatness': original_ref_metrics['spectral_flatness_diff'],
                })
                
                results.append({
                    'sample': sample_name,
                    'model': 'Fine-tuned',
                    'reference_mfcc_similarity': finetuned_ref_metrics['mfcc_similarity'],
                    'reference_spectral_contrast': finetuned_ref_metrics['spectral_contrast_diff'],
                    'reference_spectral_flatness': finetuned_ref_metrics['spectral_flatness_diff'],
                })
            
            # Compute metrics between models
            model_comparison = compute_spectral_metrics(original_audio, finetuned_audio)
            
            # Store metrics
            results.append({
                'sample': sample_name,
                'comparison': 'Between Models',
                'mfcc_similarity': model_comparison['mfcc_similarity'],
                'spectral_contrast_diff': model_comparison['spectral_contrast_diff'],
                'spectral_flatness_diff': model_comparison['spectral_flatness_diff']
            })
            
        except Exception as e:
            print(f"Error processing outputs for {midi_path.stem}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def plot_comparison_metrics(results):
    """Plot comparison metrics between models"""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Check if we have results to plot
    if df.empty:
        print("No results to plot. Skipping visualization.")
        return None
    
    # Check if 'model' column exists
    if 'model' not in df.columns:
        print("Results don't contain model comparisons. Skipping visualization.")
        return None
    
    # Filter rows with reference comparisons
    ref_df = df[df['model'].notna()]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot MFCC similarity to reference (higher is better)
    sns.barplot(x='model', y='reference_mfcc_similarity', data=ref_df, ax=axes[0])
    axes[0].set_title('MFCC Similarity to Reference Audio')
    axes[0].set_ylabel('Similarity (higher is better)')
    axes[0].set_ylim(0, 1)
    
    # Plot spectral contrast difference to reference (lower is better)
    sns.barplot(x='model', y='reference_spectral_contrast', data=ref_df, ax=axes[1])
    axes[1].set_title('Spectral Contrast Difference')
    axes[1].set_ylabel('Difference (lower is better)')
    
    # Plot spectral flatness difference to reference (lower is better)
    sns.barplot(x='model', y='reference_spectral_flatness', data=ref_df, ax=axes[2])
    axes[2].set_title('Spectral Flatness Difference')
    axes[2].set_ylabel('Difference (lower is better)')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_metrics.png", dpi=300)
    plt.close()
    
    # Create violin plots showing distribution of metrics
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='model', y='reference_mfcc_similarity', data=ref_df)
    plt.title('Distribution of MFCC Similarity to Reference Audio')
    plt.ylabel('Similarity (higher is better)')
    plt.savefig(output_dir / "mfcc_similarity_distribution.png", dpi=300)
    plt.close()
    
    # Create heatmap of metrics
    pivot_df = ref_df.pivot_table(
        index='sample', 
        columns='model', 
        values='reference_mfcc_similarity'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis')
    plt.title('MFCC Similarity to Reference by Sample and Model')
    plt.tight_layout()
    plt.savefig(output_dir / "similarity_heatmap.png", dpi=300)
    
    return fig

if __name__ == "__main__":
    print("Loading models...")
    original_model, finetuned_model, processor, device = load_models_and_processor(original_model_path, finetuned_model_path)
    
    print(f"Device: {device}")
    
    # Get list of MIDI files for testing
    midi_paths = list(Path(midi_dir).glob("**/*.mid*"))
    if not midi_paths:
        print(f"No MIDI files found in {midi_dir}")
        exit(1)
    
    print(f"Found {len(midi_paths)} MIDI files")
    
    # Set number of samples to process (limit for faster processing)
    sample_count = min(10, len(midi_paths))
    
    print(f"Generating audio and computing metrics for {sample_count} samples...")
    results = generate_audio_and_compare(
        processor, original_model, finetuned_model, midi_paths, device, sample_count
    )
    
    print("Plotting comparison metrics...")
    plot_comparison_metrics(results)
    
    print(f"Comparison complete. Results saved to {output_dir}")