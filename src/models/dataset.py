#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom dataset class for paired audio and MIDI files
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Keep this import - it's importing from data_processing.py which is fine
from data_processing import (
    preprocess_audio_file, 
    extract_notes_from_midi, 
    create_instrument_prompt
)

class AudioMIDIDataset(Dataset):
    """Dataset class for loading paired audio and MIDI files"""
    
    def __init__(
        self,
        midi_dir,
        audio_dir,
        instrument="flute",
        processor=None,
        audio_encoder=None,
        device="cpu",
        max_examples=None,
        target_sr=32000,
        max_length_seconds=10,
        require_both=False,  # If True, only use examples with both MIDI and audio
        audio_only=False,    # If True, only use audio files (ignore MIDI)
    ):
        """
        Initialize the dataset.
        
        Args:
            midi_dir: Directory containing MIDI files
            audio_dir: Directory containing audio files
            instrument: Target instrument name for prompts
            processor: MusicGen processor for tokenization (optional)
            audio_encoder: MusicGen audio encoder (optional)
            device: Device to run processing on
            max_examples: Maximum number of examples to include
            target_sr: Target sample rate for audio
            max_length_seconds: Maximum audio length in seconds
            require_both: If True, only use examples with both MIDI and audio
            audio_only: If True, only use audio files (ignore MIDI)
        """
        self.midi_dir = Path(midi_dir)
        self.audio_dir = Path(audio_dir)
        self.instrument = instrument
        self.processor = processor
        self.audio_encoder = audio_encoder
        self.device = device
        self.target_sr = target_sr
        self.max_length_seconds = max_length_seconds
        self.require_both = require_both
        self.audio_only = audio_only
        
        # Find all files
        self.examples = self._find_paired_examples(max_examples)
        
        print(f"Created dataset with {len(self.examples)} examples")
        
    def _find_paired_examples(self, max_examples=None):
        """Find paired audio and MIDI files"""
        examples = []
        
        # Get audio files
        audio_files = list(self.audio_dir.glob("*.wav"))
        if not audio_files:
            print(f"No audio files found in {self.audio_dir}")
            if not self.audio_only:
                raise ValueError("No audio files found")
                
        # For audio-only dataset
        if self.audio_only:
            if max_examples and len(audio_files) > max_examples:
                audio_files = audio_files[:max_examples]
                
            for audio_path in tqdm(audio_files, desc="Processing audio files"):
                # Create example from just audio
                piece_name = audio_path.stem.replace('_', ' ')
                examples.append({
                    "audio_path": audio_path,
                    "midi_path": None,
                    "piece_name": piece_name
                })
            return examples
        
        # For paired dataset
        midi_files = list(self.midi_dir.glob("*.mid")) + list(self.midi_dir.glob("*.midi"))
        if not midi_files:
            print(f"No MIDI files found in {self.midi_dir}")
            if self.require_both:
                raise ValueError("No MIDI files found")
        
        # Create dictionaries for lookup by stem name
        midi_dict = {midi_path.stem: midi_path for midi_path in midi_files}
        audio_dict = {audio_path.stem: audio_path for audio_path in audio_files}
        
        # Find pairs with matching names
        if self.require_both:
            # Only use examples with both MIDI and audio
            common_stems = set(midi_dict.keys()).intersection(set(audio_dict.keys()))
            for stem in common_stems:
                examples.append({
                    "audio_path": audio_dict[stem],
                    "midi_path": midi_dict[stem],
                    "piece_name": stem.replace('_', ' ')
                })
        else:
            # Use all audio files, adding MIDI if available
            for stem, audio_path in audio_dict.items():
                examples.append({
                    "audio_path": audio_path,
                    "midi_path": midi_dict.get(stem, None),
                    "piece_name": stem.replace('_', ' ')
                })
        
        # Limit examples if needed
        if max_examples and len(examples) > max_examples:
            examples = examples[:max_examples]
        
        return examples
    
    def __len__(self):
        """Return the number of examples in the dataset"""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset
        
        Returns:
            Dict with processed example data
        """
        example = self.examples[idx]
        
        # Process audio
        audio = preprocess_audio_file(
            example["audio_path"], 
            target_sr=self.target_sr,
            max_length_seconds=self.max_length_seconds
        )
        
        # Extract notes from MIDI if available
        notes = []
        if example["midi_path"]:
            notes = extract_notes_from_midi(example["midi_path"])
            
        # Create prompt
        if notes:
            # If we have notes, use them in the prompt
            notes_str = " ".join(notes[:10])
            prompt = (
                f"Generate a solo {self.instrument} performance playing {example['piece_name']}. "
                f"The piece uses these notes: {notes_str}... "
                f"The {self.instrument} should play with proper breath control, fluid phrasing, "
                f"and accurate timing."
            )
        else:
            # Otherwise, create a standard prompt
            prompt = create_instrument_prompt(example["piece_name"], self.instrument)
        
        # Create basic return dict
        result = {
            "prompt": prompt,
            "audio": audio,
            "audio_path": str(example["audio_path"]),
            "piece_name": example["piece_name"]
        }
        
        if example["midi_path"]:
            result["midi_path"] = str(example["midi_path"])
            result["notes"] = notes
            
        # If processor and encoder are available, preprocess for model
        if self.processor and self.audio_encoder:
            # Process text prompt
            text_inputs = self.processor.tokenizer(
                prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            # Process audio
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            audio_inputs = self.processor.feature_extractor(
                audio_tensor, 
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            
            # Encode audio with the encoder
            with torch.no_grad():
                # Add batch dimension if needed
                if len(audio_inputs.input_values.shape) == 1:
                    audio_val = audio_inputs.input_values.unsqueeze(0)
                else:
                    audio_val = audio_inputs.input_values
                    
                # Get audio codes
                encoded_audio = self.audio_encoder(audio_val.to(self.device))
                
                # Extract tokens based on shape
                if hasattr(encoded_audio, "audio_codes"):
                    audio_codes = encoded_audio.audio_codes
                    
                    # Handle different shapes
                    if len(audio_codes.shape) == 4:  # [1, 1, codebooks, seq_len]
                        labels = audio_codes[0, 0, 0, :].clone().cpu()
                    elif len(audio_codes.shape) == 3:  # [batch, codebooks, seq_len]
                        labels = audio_codes[0, 0, :].clone().cpu()
                    else:
                        # Fallback
                        labels = audio_codes.view(-1).cpu()
                else:
                    # Fallback if no audio codes
                    labels = torch.zeros(500, dtype=torch.long)
            
            # Add model-ready tensors
            result["input_ids"] = text_inputs.input_ids.squeeze(0)
            result["attention_mask"] = text_inputs.attention_mask.squeeze(0)
            result["labels"] = labels
            
        return result

def create_data_loaders(
    midi_dir,
    audio_dir,
    instrument="flute",
    processor=None,
    audio_encoder=None,
    device="cpu",
    batch_size=1,
    train_split=0.8,
    shuffle=True,
    max_examples=None
):
    """
    Create train and validation data loaders
    
    Args:
        midi_dir: Directory containing MIDI files
        audio_dir: Directory containing audio files
        instrument: Target instrument name
        processor: MusicGen processor
        audio_encoder: MusicGen audio encoder
        device: Device to run processing on
        batch_size: Batch size for training
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data
        max_examples: Maximum number of examples to use
        
    Returns:
        train_loader, val_loader
    """
    # Create dataset
    dataset = AudioMIDIDataset(
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        instrument=instrument,
        processor=processor,
        audio_encoder=audio_encoder,
        device=device,
        max_examples=max_examples
    )
    
    # Split into train and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

def custom_collate_fn(batch):
    """
    Custom collation function to handle variable-length sequences
    
    Args:
        batch: List of examples from the dataset
        
    Returns:
        Dict with batched tensors
    """
    # Get keys from first batch
    keys = batch[0].keys()
    
    # Keys that should be stacked as tensors
    tensor_keys = ["input_ids", "attention_mask", "labels"]
    
    # Create result dict
    result = {}
    
    for key in keys:
        if key in tensor_keys and key in batch[0] and isinstance(batch[0][key], torch.Tensor):
            # Stack tensors
            try:
                result[key] = torch.stack([example[key] for example in batch])
            except Exception as e:
                print(f"Error stacking {key}: {e}")
                # Fallback - use first example's shape to create zeros
                shape = batch[0][key].shape
                result[key] = torch.zeros((len(batch), *shape), dtype=batch[0][key].dtype)
        else:
            # Keep as list for non-tensor values
            result[key] = [example.get(key) for example in batch]
    
    return result