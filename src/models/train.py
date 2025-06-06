#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge transfer training for MusicGen to generate instrumental performances from MIDI input
"""

import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import librosa
import mido
import json
import tempfile
import boto3
import shutil
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split

from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    get_linear_schedule_with_warmup
)

# For MIDI processing
from pretty_midi import PrettyMIDI
from miditoolkit import MidiFile

# S3 Configuration
S3_CONFIG = {
    'bucket': 'iit-symphoniq',
    'aws_access_key_id': 'AKIAQUFLP4IVKRLIXANY',
    'aws_secret_access_key': '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',
    'region_name': 'ap-south-1',
    'default_prefix': 'uploads/'  # Default prefix for uploads
}

class S3FileHandler:
    """Handles S3 file operations and caching"""
    
    def __init__(self, config):
        self.config = config
        self.bucket = config['bucket']
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key'],
            region_name=config['region_name']
        )
        # Create temp directory for downloaded files
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = {}  # Maps S3 keys to local paths
        
    def __del__(self):
        # Clean up temp files when instance is destroyed
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def list_files(self, prefix, extension=None):
        """List files in S3 bucket with given prefix and optional extension"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            
            files = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if extension is None or key.lower().endswith(extension.lower()):
                            files.append(key)
            return files
        except ClientError as e:
            print(f"Error listing S3 objects: {e}")
            return []
    
    def download_file(self, s3_key):
        """Download file from S3 and return local path"""
        if s3_key in self.cache:
            return self.cache[s3_key]
        
        local_path = self.temp_dir / Path(s3_key).name
        try:
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))
            self.cache[s3_key] = local_path
            return local_path
        except ClientError as e:
            print(f"Error downloading {s3_key}: {e}")
            return None
    
    def upload_file(self, local_path, s3_key):
        """Upload file to S3 bucket"""
        try:
            self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
            return True
        except ClientError as e:
            print(f"Error uploading to {s3_key}: {e}")
            return False
    def ensure_local_file(self, path):
        """Ensures a file is available locally, downloading if needed"""
        if not isinstance(path, str):
            return path
        
        # If path is already local, return it
        if os.path.exists(path):
            return path
            
        # Try to download from S3
        local_path = self.download_file(path)
        if local_path:
            print(f"✓ Downloaded {os.path.basename(path)} from S3")
            return local_path
        else:
            print(f"✗ Failed to download {path} from S3")
            return None

# Dataset class for paired MIDI-audio samples
class MIDIAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        midi_dir,
        audio_dir,
        processor,
        audio_encoder,
        device="cpu",
        max_examples=None,
        target_sr=32000,
        max_length_seconds=10,
        midi_representation="text",  # "text", "piano_roll", or "events"
        train_mode=True,
        test_split=0.1,
        s3_handler=None 
    ):
        self.processor = processor
        self.audio_encoder = audio_encoder
        self.device = device
        self.target_sr = target_sr
        self.max_length = int(target_sr * max_length_seconds)
        self.midi_representation = midi_representation
        self.train_mode = train_mode
        self.s3_handler = s3_handler
        
        # Find all MIDI and audio files
        if s3_handler:
            # Using S3 storage
            midi_files = sorted(s3_handler.list_files(midi_dir, extension=".mid"))
            audio_files = sorted(s3_handler.list_files(audio_dir, extension=".wav"))
            
            print(f"Found {len(midi_files)} MIDI files and {len(audio_files)} audio files in S3")
        else:
            # Using local storage
            midi_dir = Path(midi_dir)
            midi_files = sorted(list(midi_dir.glob("**/*.mid")))
            
            audio_dir = Path(audio_dir)
            audio_files = sorted(list(audio_dir.glob("**/*.wav")))
            
            print(f"Found {len(midi_files)} MIDI files and {len(audio_files)} audio files locally")
        
        
        # Create paired examples
        self.examples = []
        
        print(f"Found {len(midi_files)} MIDI files and {len(audio_files)} audio files")
        
        # Match MIDI and audio files by name
                # Match MIDI and audio files by name
        midi_dict = {}
        for f in midi_files:
            # Handle both Path objects and strings (S3 keys)
            if isinstance(f, str):
                # For S3 paths
                filename = os.path.basename(f)
                stem = os.path.splitext(filename)[0]
            else:
                # For Path objects
                stem = f.stem
                
            # Try different ways to get a base name
            base_name = stem.split('_')[0]  # Original way
            midi_dict[base_name] = f
            
            # Also try the full stem as a key
            midi_dict[stem] = f
        
        audio_dict = {}
        for f in audio_files:
            # Handle both Path objects and strings (S3 keys)
            if isinstance(f, str):
                # For S3 paths
                filename = os.path.basename(f)
                stem = os.path.splitext(filename)[0]
            else:
                # For Path objects
                stem = f.stem
                
            # Try different ways to get a base name
            base_name = stem.split('_')[0]  # Original way
            audio_dict[base_name] = f
            
            # Also try the full stem as a key
            audio_dict[stem] = f
        
        # Find common keys
        common_keys = set(midi_dict.keys()) & set(audio_dict.keys())
        print(f"Found {len(common_keys)} paired examples")
        
        # Create examples
        for key in common_keys:
            self.examples.append({
                "midi_path": str(midi_dict[key]),
                "audio_path": str(audio_dict[key])
            })
        
        # Limit examples if needed
        if max_examples is not None and max_examples < len(self.examples):
            self.examples = self.examples[:max_examples]
        
        
        if len(self.examples) < 10:
            print("Few pairs found with exact matching, trying fuzzy matching...")
            
            # Try matching based on substring presence
            additional_pairs = []
            for midi_file in midi_files:
                # Get the stem of the file, handling both Path and string
                if isinstance(midi_file, str):
                    midi_filename = os.path.basename(midi_file)
                    midi_name = os.path.splitext(midi_filename)[0].lower()
                else:
                    midi_name = midi_file.stem.lower()
                
                for audio_file in audio_files:
                    # Get the stem of the file, handling both Path and string
                    if isinstance(audio_file, str):
                        audio_filename = os.path.basename(audio_file)  
                        audio_name = os.path.splitext(audio_filename)[0].lower()
                    else:
                        audio_name = audio_file.stem.lower()
                    
                    # Check if one name is a substring of the other 
                    # or they share a common substring of length >= 5
                    if (midi_name in audio_name or audio_name in midi_name or
                        any(len(substr) >= 5 for substr in set(midi_name.split('_')) & set(audio_name.split('_')))):
                        additional_pairs.append({
                            "midi_path": str(midi_file),
                            "audio_path": str(audio_file)
                        })
            
            print(f"Found {len(additional_pairs)} additional pairs with fuzzy matching")
            self.examples.extend(additional_pairs)
            
            # Remove duplicates if any
            seen = set()
            self.examples = [x for x in self.examples if not (tuple(x.items()) in seen or seen.add(tuple(x.items())))]
        
        
        # Split into train and test
        if test_split > 0:
            train_examples, test_examples = train_test_split(
                self.examples, test_size=test_split, random_state=42
            )
            self.examples = train_examples if train_mode else test_examples
        
        print(f"Dataset contains {len(self.examples)} examples for {'training' if train_mode else 'testing'}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        midi_path = example["midi_path"]
        audio_path = example["audio_path"]
        
        # Process MIDI file
        midi_prompt = self.process_midi(midi_path)
        
        # Process audio file
        audio = self.process_audio(audio_path)
        
        # Process prompt text
        inputs = self.processor(text=[midi_prompt], padding=True, return_tensors="pt")
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        
        # Process audio for labels
        with torch.no_grad():
            audio_torch = torch.tensor(audio).to(self.device)
            # Add both batch and channel dimensions (EnCodec expects [batch, channels, time])
            audio_torch = audio_torch.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, time]
            
            # Extract audio features using the model's audio encoder
            encodec_output = self.audio_encoder(audio_torch)
            
            # Get the audio codes from the EncodecOutput object
            # EncodecOutput has audio_codes attribute rather than being a tensor
            if hasattr(encodec_output, 'audio_codes'):
                labels = encodec_output.audio_codes.squeeze(0)
            elif hasattr(encodec_output, 'audio_values'):
                labels = encodec_output.audio_values.squeeze(0)
            else:
                # Fallback for different versions of the encoder
                try:
                    labels = encodec_output[0].squeeze(0)
                except:
                    raise ValueError(f"Cannot extract audio values from encoder output: {type(encodec_output)}")
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio": audio,
            "midi_path": midi_path,
            "audio_path": audio_path,
            "prompt": midi_prompt
        }
    
    def process_audio(self, audio_path):
        """Load and process audio file"""
        # Handle S3 paths
        if self.s3_handler and isinstance(audio_path, str) and not os.path.exists(audio_path):
            # Download from S3 to local temp path
            local_path = self.s3_handler.download_file(audio_path)
            if local_path is None:
                raise ValueError(f"Could not download audio file: {audio_path}")
            audio_path = local_path
        
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
    
        return audio
    
    def process_midi(self, midi_path):
        """Convert MIDI to text representation"""
        try:
            # Handle S3 paths
            if self.s3_handler and isinstance(midi_path, str) and not os.path.exists(midi_path):
                # Download from S3 to local temp path
                local_path = self.s3_handler.download_file(midi_path)
                if local_path is None:
                    raise ValueError(f"Could not download MIDI file: {midi_path}")
                midi_path = local_path
                
            if self.midi_representation == "text":
                # Parse MIDI and convert to text description
                midi_data = PrettyMIDI(midi_path)
                
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
                
                # Fix: Get tempo safely using get_tempo_changes() method
                tempo = 120  # Default
                try:
                    _, tempos = midi_data.get_tempo_changes()
                    if len(tempos) > 0:
                        tempo = int(tempos[0])
                except (AttributeError, ValueError):
                    # Fallback method if get_tempo_changes fails
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
                            
            elif self.midi_representation == "piano_roll":
                # Convert MIDI to piano roll and then to text description
                midi_data = PrettyMIDI(midi_path)
                piano_roll = midi_data.get_piano_roll(fs=16)  # 16 frames per second
                active_notes = np.sum(piano_roll > 0, axis=0)
                mean_pitch = np.sum(np.arange(128).reshape(-1, 1) * (piano_roll > 0), axis=0)
                mean_pitch = np.divide(mean_pitch, active_notes, out=np.zeros_like(mean_pitch), where=active_notes > 0)
                
                # Create blocks of midi content
                blocks = []
                for i in range(0, len(active_notes), 16):
                    notes_in_block = int(np.sum(active_notes[i:i+16] > 0))
                    if notes_in_block > 0:
                        avg_pitch_in_block = int(np.mean(mean_pitch[i:i+16][active_notes[i:i+16] > 0]))
                        blocks.append(f"{notes_in_block} notes near {librosa.midi_to_note(avg_pitch_in_block)}")
                
                blocks_text = ", then ".join(blocks[:10])
                if len(blocks) > 10:
                    blocks_text += f", and {len(blocks) - 10} more segments"
                
                prompt = (
                    f"Create a flute performance of this musical piece. "
                    f"The melody structure is: {blocks_text}. "
                    f"The flute should play expressively with clear articulation and good breath control."
                )
                
                return prompt
                
            else:
                # Default simple text representation
                mid = MidiFile(midi_path)
                
                # Extract track and note count information
                num_tracks = len(mid.instruments)
                total_notes = sum(len(instr.notes) for instr in mid.instruments)
                
                prompt = (
                    f"Create a flute performance based on a MIDI file with {num_tracks} tracks "
                    f"and {total_notes} notes. The flute should play expressively with "
                    f"clear articulation and good breath control."
                )
                
                return prompt
                
        except Exception as e:
            print(f"Error processing MIDI {midi_path}: {e}")
            return "Create a flute performance based on a MIDI file. The flute should play expressively."


# Custom collate function
def custom_collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    audio = [item["audio"] for item in batch]
    midi_paths = [item["midi_path"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "audio": audio,
        "midi_paths": midi_paths,
        "audio_paths": audio_paths,
        "prompts": prompts
    }


def knowledge_transfer_train(
    teacher_model_path="/Users/gauravs/Documents/Symphoniq/src/models/cached_models",
    student_model_name="facebook/musicgen-small",
    midi_dir="/Users/gauravs/Documents/Symphoniq/src/data/augmented/midi",
    audio_dir="/Users/gauravs/Documents/Symphoniq/src/data/augmented/audio",
    output_dir="fine_tuned_model",
    batch_size=1,
    epochs=5,
    learning_rate=1e-5,
    alpha=0.5,
    temperature=1.0,
    warmup_steps=100,
    save_every=100,
    eval_every=50,
    max_examples=None,
    midi_representation="text",
    use_s3=True,  # Changed default to True
    s3_config=None,
    project_name=None  # Added project name parameter
):
    """
    Train a student MusicGen model using knowledge transfer from a teacher model
    
    Args:
        teacher_model_path: Local path to the cached teacher model
        student_model_name: Hugging Face model name for student
        midi_dir: Directory containing MIDI files
        audio_dir: Directory containing audio files
        output_dir: Directory to save the model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        alpha: Weight for knowledge transfer loss (0-1)
        temperature: Temperature for knowledge distillation
        warmup_steps: Number of warmup steps for scheduler
        save_every: Save checkpoint every N steps
        eval_every: Evaluate every N steps
        max_examples: Maximum number of examples to use
        midi_representation: How to represent MIDI ("text", "piano_roll", "events")
        use_s3: Whether to use S3 storage
        s3_config: S3 configuration dictionary
        project_name: Name of the project/instrument (e.g., "flute", "violin")
    """
    # If project name is provided, customize the prompts
    instrument_name = project_name if project_name else "flute"
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    s3_handler = None
    if use_s3 and s3_config:
        print(f"Initializing S3 handler with bucket: {s3_config['bucket']}")
        s3_handler = S3FileHandler(s3_config)
        
        # When using S3, check if the directories exist, if not create default paths
        if not s3_handler.list_files(midi_dir):
            print(f"Warning: MIDI directory {midi_dir} not found or empty in S3")
            # Try a default location
            default_midi_dir = f"uploads/{project_name}/midi" if project_name else "uploads/midi"
            if s3_handler.list_files(default_midi_dir):
                print(f"Using default MIDI directory: {default_midi_dir}")
                midi_dir = default_midi_dir
        
        if not s3_handler.list_files(audio_dir):
            print(f"Warning: Audio directory {audio_dir} not found or empty in S3")
            # Try a default location
            default_audio_dir = f"uploads/{project_name}/audio" if project_name else "uploads/audio"
            if s3_handler.list_files(default_audio_dir):
                print(f"Using default audio directory: {default_audio_dir}")
                audio_dir = default_audio_dir
    
    # Set device - use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load teacher model and processor from cached path
    print(f"Loading teacher model from: {teacher_model_path}")
    processor = AutoProcessor.from_pretrained(teacher_model_path)
    
    # Fix the decoder_start_token_id issue
    teacher_config = MusicgenForConditionalGeneration.config_class.from_pretrained(teacher_model_path)
    teacher_config.decoder_start_token_id = 0
    teacher_model = MusicgenForConditionalGeneration.from_pretrained(teacher_model_path, config=teacher_config)
    teacher_model.to(device)
    
    # Load student model
    print(f"Loading student model: {student_model_name}")
    student_config = MusicgenForConditionalGeneration.config_class.from_pretrained(student_model_name)
    student_config.decoder_start_token_id = 0
    student_model = MusicgenForConditionalGeneration.from_pretrained(student_model_name, config=student_config)
    student_model.to(device)
    
    # Freeze teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # Create dataset with modified prompt function that uses the project name
    print(f"Creating dataset from MIDI: {midi_dir} and Audio: {audio_dir}")
    
    # Original MIDIAudioDataset instantiations remain, but now the dataset will use the instrument name
    # via a monkey patch to make process_midi use the correct instrument name
    original_process_midi = MIDIAudioDataset.process_midi
    
    def customized_process_midi(self, midi_path):
        """Monkey-patched method to use the correct instrument name"""
        try:
            if self.midi_representation == "text":
                # Parse MIDI and convert to text description
                midi_data = PrettyMIDI(midi_path)
                
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
                
                # Fix: Get tempo safely using get_tempo_changes() method
                tempo = 120  # Default
                try:
                    _, tempos = midi_data.get_tempo_changes()
                    if len(tempos) > 0:
                        tempo = int(tempos[0])
                except (AttributeError, ValueError):
                    # Fallback method if get_tempo_changes fails
                    pass
                
                # Create text prompt using the provided instrument name
                instrument_text = ", ".join(instruments[:3]) if instruments else "Piano"
                prompt = (
                    f"Create a {instrument_name} performance based on this MIDI. "
                    f"The original is for {instrument_text}, with {num_notes} notes "
                    f"ranging from {librosa.midi_to_note(pitch_range[0])} to "
                    f"{librosa.midi_to_note(pitch_range[1])}, in {time_sig} at {tempo} BPM, "
                    f"lasting {total_time} seconds. "
                    f"The {instrument_name} should play expressively with clear articulation and good breath control."
                )
                
                return prompt
                            
            elif self.midi_representation == "piano_roll":
                # Convert MIDI to piano roll and then to text description
                midi_data = PrettyMIDI(midi_path)
                piano_roll = midi_data.get_piano_roll(fs=16)  # 16 frames per second
                active_notes = np.sum(piano_roll > 0, axis=0)
                mean_pitch = np.sum(np.arange(128).reshape(-1, 1) * (piano_roll > 0), axis=0)
                mean_pitch = np.divide(mean_pitch, active_notes, out=np.zeros_like(mean_pitch), where=active_notes > 0)
                
                # Create blocks of midi content
                blocks = []
                for i in range(0, len(active_notes), 16):
                    notes_in_block = int(np.sum(active_notes[i:i+16] > 0))
                    if notes_in_block > 0:
                        avg_pitch_in_block = int(np.mean(mean_pitch[i:i+16][active_notes[i:i+16] > 0]))
                        blocks.append(f"{notes_in_block} notes near {librosa.midi_to_note(avg_pitch_in_block)}")
                
                blocks_text = ", then ".join(blocks[:10])
                if len(blocks) > 10:
                    blocks_text += f", and {len(blocks) - 10} more segments"
                
                prompt = (
                    f"Create a {instrument_name} performance of this musical piece. "
                    f"The melody structure is: {blocks_text}. "
                    f"The {instrument_name} should play expressively with clear articulation and good breath control."
                )
                
                return prompt
                
            else:
                # Default simple text representation
                mid = MidiFile(midi_path)
                
                # Extract track and note count information
                num_tracks = len(mid.instruments)
                total_notes = sum(len(instr.notes) for instr in mid.instruments)
                
                prompt = (
                    f"Create a {instrument_name} performance based on a MIDI file with {num_tracks} tracks "
                    f"and {total_notes} notes. The {instrument_name} should play expressively with "
                    f"clear articulation and good breath control."
                )
                
                return prompt
                
        except Exception as e:
            print(f"Error processing MIDI {midi_path}: {e}")
            return f"Create a {instrument_name} performance based on a MIDI file. The {instrument_name} should play expressively."
    
    # Apply the monkey patch
    MIDIAudioDataset.process_midi = customized_process_midi
    
    train_dataset = MIDIAudioDataset(
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        processor=processor,
        audio_encoder=teacher_model.audio_encoder,
        device=device,
        max_examples=max_examples,
        target_sr=32000,
        max_length_seconds=10,
        midi_representation=midi_representation,
        train_mode=True,
        s3_handler=s3_handler
    )
    
    # Create validation dataset
    val_dataset = MIDIAudioDataset(
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        processor=processor,
        audio_encoder=teacher_model.audio_encoder,
        device=device,
        max_examples=max_examples,
        target_sr=32000,
        max_length_seconds=10,
        midi_representation=midi_representation,
        train_mode=False,
        s3_handler=s3_handler
    )
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0  # Increase if you have more CPU cores
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # Disable codebook stacking for training stability
    if hasattr(student_model.config, "use_all_codebooks"):
        student_model.config.use_all_codebooks = False
        print("Disabled multi-codebook usage for training stability")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Knowledge distillation loss function (KL divergence)
    def distillation_loss(student_logits, teacher_logits, temperature=1.0):
        soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss * (temperature ** 2)
    
    # Training loop
    global_step = 0
    train_losses = []
    best_val_loss = float('inf')
    
    # Save training config
    config = {
        "teacher_model": teacher_model_path,  # Changed from teacher_model_name to teacher_model_path
        "student_model": student_model_name,
        "midi_dir": str(midi_dir),
        "audio_dir": str(audio_dir),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "temperature": temperature,
        "warmup_steps": warmup_steps,
        "midi_representation": midi_representation,
        "max_examples": max_examples
    }
    
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Starting knowledge transfer training for {epochs} epochs...")
    for epoch in range(epochs):
        student_model.train()
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move all tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Reshape labels for MusicGen's codebook expectation (4 codebooks)
            print(f"Original labels shape: {labels.shape}")
            print(f"Original labels shape: {labels.shape}")
            if len(labels.shape) == 2:
                # If 2D: [batch_size, seq_len]
                batch_size, seq_len = labels.shape
                labels = labels.reshape(batch_size, 1, seq_len).repeat(1, 4, 1)
            elif len(labels.shape) == 3:
                # If 3D: [batch_size, num_codebooks, seq_len] - already correct format
                pass
            elif len(labels.shape) == 4:
                # If 4D: [batch_size, 1, num_codebooks, seq_len] or [batch_size, num_codebooks, frames, codebook_dim]
                # We need to reshape to [batch_size, num_codebooks, seq_len]
                if labels.shape[1] == 1:  # Shape is [batch, 1, codebooks, seq]
                    batch_size, _, num_codebooks, seq_len = labels.shape
                    labels = labels.reshape(batch_size, num_codebooks, seq_len)
                else:  # Shape is [batch, codebooks, frames, dim]
                    batch_size, num_codebooks, frames, _ = labels.shape
                    labels = labels.reshape(batch_size, num_codebooks, -1)
            else:
                # Handle other unexpected shapes
                raise ValueError(f"Unexpected labels shape: {labels.shape}. Expected 2D, 3D, or 4D tensor.")
            print(f"Reshaped labels shape: {labels.shape}")
            
            # Forward pass - student
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
            
            # Standard student loss
            student_loss = student_outputs.loss
            
            # Forward pass - teacher (for knowledge transfer)
            with torch.no_grad():
                # Call just the text encoder part of the teacher model to avoid audio encoding errors
                teacher_text_outputs = teacher_model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get the text encoder hidden states instead of the full model's hidden states
                teacher_hidden = teacher_text_outputs.hidden_states[-1]
            
            # Knowledge transfer loss - align hidden states
            distill_loss = 0
            
            # Using last hidden state from both models
            student_text_outputs = student_outputs.encoder_hidden_states
            student_hidden = student_text_outputs[-1] if student_text_outputs else student_outputs.hidden_states[-1]
                # If different dimensions, project student hidden states
            if student_hidden.shape != teacher_hidden.shape:
                # Simple mean pooling if shapes don't match
                if student_hidden.shape[-1] < teacher_hidden.shape[-1]:
                    # Project student hidden state up to teacher dimension
                    student_hidden = torch.nn.functional.linear(
                        student_hidden, 
                        torch.randn(teacher_hidden.shape[-1], student_hidden.shape[-1], device=device)
                    )
                else:
                    # Project teacher hidden state up to student dimension
                    teacher_hidden = torch.nn.functional.linear(
                        teacher_hidden,
                        torch.randn(student_hidden.shape[-1], teacher_hidden.shape[-1], device=device)
                    )
            
            # Calculate MSE between hidden states
            distill_loss = torch.nn.functional.mse_loss(
                student_hidden, teacher_hidden
            )
            
            # Calculate final loss
            loss = (1 - alpha) * student_loss + alpha * distill_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update progress
            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}", 
                student=f"{student_loss.item():.4f}",
                distill=f"{distill_loss.item():.4f}"
            )
            
            # Save checkpoint
            global_step += 1
            if global_step % save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                student_model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
            
            # Evaluate
            if global_step % eval_every == 0:
                # Validation
                student_model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc="Validating"):
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        
                        # Reshape labels for MusicGen's codebook expectation
                        if len(val_labels.shape) == 2:
                            val_batch_size, val_seq_len = val_labels.shape
                            val_labels = val_labels.reshape(val_batch_size, 1, val_seq_len).repeat(1, 4, 1)
                        elif len(val_labels.shape) == 4 and val_labels.shape[1] == 1:
                            # 4D shape with dimension structure [batch, 1, codebooks, seq]
                            val_batch_size, _, val_num_codebooks, val_seq_len = val_labels.shape
                            val_labels = val_labels.reshape(val_batch_size, val_num_codebooks, val_seq_len)
                        elif len(val_labels.shape) == 4:
                            # Other 4D shape
                            val_batch_size, val_num_codebooks, val_frames, _ = val_labels.shape
                            val_labels = val_labels.reshape(val_batch_size, val_num_codebooks, -1)
                        
                        # Forward pass
                        val_outputs = student_model(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask,
                            labels=val_labels
                        )
                        
                        val_losses.append(val_outputs.loss.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
                print(f"Validation loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = output_dir / "best_model"
                    student_model.save_pretrained(best_model_path)
                    processor.save_pretrained(best_model_path)
                    print(f"New best model saved! Validation loss: {best_val_loss:.4f}")
                
                # Generate a sample for qualitative evaluation
                midi_path = val_dataset[0]["midi_path"] if len(val_dataset) > 0 else train_dataset[0]["midi_path"]
                # Ensure we're using a local path, not an S3 path
                if s3_handler and isinstance(midi_path, str) and not os.path.exists(midi_path):
                    local_midi_path = s3_handler.download_file(midi_path)
                    if local_midi_path:
                        midi_path = local_midi_path
                    else:
                        print(f"Warning: Could not download {midi_path} for sample generation")
                        # Use a fallback prompt if download fails
                        prompt = f"Create a {instrument_name} performance based on a MIDI file."
                        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                else:
                    prompt = train_dataset.process_midi(midi_path)  # Now safely handles the file
                    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                # Generate audio
                audio_values = student_model.generate(
                    **inputs,
                    max_new_tokens=500,
                    guidance_scale=3.0,
                    temperature=0.7
                ).cpu().numpy()
            
                # Save generated audio
                audio_path = output_dir / f"{project_name}_sample_{global_step}.wav" if project_name else output_dir / f"sample_{global_step}.wav"
                try:
                    import scipy.io.wavfile
                    scipy.io.wavfile.write(
                        audio_path, 
                        rate=32000, 
                        data=(audio_values[0] * 32767).astype(np.int16)
                    )
                    
                    # Also save the prompt
                    with open(output_dir / f"sample_{global_step}_prompt.txt", "w") as f:
                        f.write(prompt)
                    
                    print(f"Saved sample audio to {audio_path}")
                except Exception as e:
                    print(f"Error saving audio sample: {str(e)}")
                
                # Switch back to train mode
                student_model.train()
        
        # End of epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs} complete. Average loss: {avg_loss:.4f}")
    
    # Save final model
    final_model_path = output_dir / "final_model"
    student_model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Training complete! Model saved to {final_model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(output_dir / "training_loss.png")
    
    return final_model_path


if __name__ == "__main__":
    # Fix for musicgen's shift_tokens_right function
    import transformers.models.musicgen.modeling_musicgen as musicgen_module
    
    # Safe implementation that doesn't fail on missing decoder_start_token_id
    def safe_shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id=None):
        """Shift input ids one token to the right, and wrap the last non-padding token."""
        if decoder_start_token_id is None:
            decoder_start_token_id = 0
            print(f"Using default decoder_start_token_id={decoder_start_token_id}")
            
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        
        # Replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        
        return shifted_input_ids
    
    # Apply the patch
    musicgen_module.shift_tokens_right = safe_shift_tokens_right
    print("Applied global patch for shift_tokens_right function")
    
    # Patch the from_pretrained method to properly handle local files
    original_processor_from = AutoProcessor.from_pretrained
    def patched_from_pretrained(pretrained_path, *args, **kwargs):
        if os.path.exists(pretrained_path):
            kwargs['local_files_only'] = True 
            kwargs['trust_remote_code'] = True
        return original_processor_from(pretrained_path, *args, **kwargs)
    AutoProcessor.from_pretrained = patched_from_pretrained

    # Also patch MusicgenForConditionalGeneration
    original_model_from = MusicgenForConditionalGeneration.from_pretrained
    def patched_model_from_pretrained(pretrained_path, *args, **kwargs):
        if os.path.exists(pretrained_path):
            kwargs['local_files_only'] = True
            kwargs['trust_remote_code'] = True
        return original_model_from(pretrained_path, *args, **kwargs)
    MusicgenForConditionalGeneration.from_pretrained = patched_model_from_pretrained

    # Parse arguments - simplified to take just a project name
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MusicGen with knowledge transfer for MIDI-to-instrument conversion")
    parser.add_argument("project_name", type=str, help="Name of the instrument/project (e.g., 'flute', 'violin')")
    parser.add_argument("--teacher", type=str, default="/home/cc/Symphoniq/src/models/cached_models", 
                       help="Path to cached teacher model")
    parser.add_argument("--student", type=str, default="facebook/musicgen-small", help="Student model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Derive S3 directories from project name
    project_name = args.project_name
    midi_dir = f"uploads/{project_name}/midi"
    audio_dir = f"uploads/{project_name}/audio"
    output_dir = f"{project_name}_model"
    
    print(f"Starting training for {project_name}...")
    print(f"Looking for MIDI files in S3 path: {midi_dir}")
    print(f"Looking for audio files in S3 path: {audio_dir}")
    print(f"Model outputs will be saved to: {output_dir}")
    
    # Start training with knowledge transfer
    knowledge_transfer_train(
        teacher_model_path=args.teacher,
        student_model_name=args.student,
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        output_dir=output_dir,
        batch_size=1,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha=0.5,
        temperature=1.0,
        midi_representation="text",
        use_s3=True,
        s3_config=S3_CONFIG,
        project_name=project_name
    )
    
    # Fix direct S3 access issue - Add this new code
    import io
    import soundfile as sf
    
    def get_local_path(s3_handler, path):
        """Ensures a path is available locally, downloading if needed"""
        if not isinstance(path, str):
            return str(path)
            
        # If already local
        if os.path.exists(path):
            return path
            
        # If it's an S3 path
        if '/' in path and not os.path.exists(os.path.dirname(path)):
            local_path = s3_handler.download_file(path)
            if local_path:
                return str(local_path)
                
        # Return original if we couldn't handle it
        return path
    
    # Monkey patch MIDIAudioDataset.__getitem__ to properly handle S3 paths
    original_getitem = MIDIAudioDataset.__getitem__
    
    def patched_getitem(self, idx):
        example = self.examples[idx]
        if self.s3_handler:
            # Preemptively try to download the files before processing
            for key in ['midi_path', 'audio_path']:
                if key in example and isinstance(example[key], str):
                    if not os.path.exists(example[key]):
                        local_path = self.s3_handler.download_file(example[key])
                        if local_path:
                            example[key] = str(local_path)
        
        # Call original method
        return original_getitem(self, idx)
    
    # Apply the patch
    MIDIAudioDataset.__getitem__ = patched_getitem
    
    # Also patch the process_midi function to better handle S3 paths
    original_process_midi = MIDIAudioDataset.process_midi
    
    def patched_process_midi(self, midi_path):
        try:
            # Handle S3 paths more aggressively
            if self.s3_handler and isinstance(midi_path, str) and not os.path.exists(midi_path):
                local_path = self.s3_handler.download_file(midi_path)
                if local_path is not None:
                    midi_path = str(local_path)
                else:
                    print(f"Warning: Could not download MIDI: {midi_path}, using fallback prompt")
                    return f"Create a performance based on a MIDI file. Play expressively."
            
            return original_process_midi(self, midi_path)
        except Exception as e:
            print(f"Error in patched process_midi for {midi_path}: {e}")
            return f"Create a performance based on a MIDI file. Play expressively."
    
    # Apply the MIDI processing patch
    MIDIAudioDataset.process_midi = patched_process_midi
    
    # Parse arguments...