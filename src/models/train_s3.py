#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import librosa
import json
import io
import boto3
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
import random
import torch.nn.functional as F

from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import torch
import gc

# Force garbage collection and clear CUDA cache
gc.collect()
torch.cuda.empty_cache()

# Disable CUDA memory caching which is causing the NVML error
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
# For MIDI processing
from pretty_midi import PrettyMIDI
import soundfile as sf

# S3 Configuration
S3_CONFIG = {
    'bucket': 'iit-symphoniq',
    'aws_access_key_id': 'AKIAQUFLP4IVKRLIXANY',
    'aws_secret_access_key': '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',
    'region_name': 'ap-south-1'
}

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_CONFIG['aws_access_key_id'],
    aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
    region_name=S3_CONFIG['region_name']
)
bucket_name = S3_CONFIG['bucket']

# In-memory cache for S3 objects
data_cache = {}  # Maps S3 keys to parsed data

def list_s3_files(prefix, extension=None):
    """List files in S3 bucket with given prefix and optional extension"""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
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

class SpectralNormalization:
    """
    Weight Normalization Technique
    - Purpose: Stabilize training by constraining weight matrices
    - Method: Normalize weights by their spectral norm
    - Effect: Prevents sudden changes in model behavior
    """
    @staticmethod
    def apply(model, name='weight', n_power_iterations=1):
        for module in model.modules():
            if hasattr(module, name):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                    weight = getattr(module, name)
                    with torch.no_grad():
                        # Calculate spectral norm and normalize
                        weight_mat = weight.view(weight.size(0), -1)
                        u = torch.randn(weight_mat.size(0), device=weight.device)
                        
                        # Power iteration
                        for _ in range(n_power_iterations):
                            v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0)
                            u = F.normalize(torch.matmul(weight_mat, v), dim=0)
                        
                        # Spectral norm
                        sigma = torch.dot(u, torch.matmul(weight_mat, v))
                        
                        # Normalize weight
                        if sigma > 0:
                            module.weight.data = weight / sigma

class FocalLoss(torch.nn.Module):
    """
    Advanced Loss Function
    - Purpose: Better handle imbalanced data
    - Method: Down-weight easy examples, focus on hard ones
    - Effect: More robust training on diverse audio patterns
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def mixup_data(x, y, alpha=0.2):
    """
    Data Augmentation Technique
    - Purpose: Create new training examples
    - Method: Linear interpolation of inputs and labels
    - Effect: Smoother decision boundaries, better generalization
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Audio augmentation
def augment_audio(audio, target_sr, max_pitch_shift=2, max_speed=0.1):
    """Apply random augmentation to audio"""
    if random.random() < 0.5:
        # Random pitch shift between -max_pitch_shift and max_pitch_shift semitones
        n_steps = random.uniform(-max_pitch_shift, max_pitch_shift)
        audio = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=n_steps)
    
    if random.random() < 0.5:
        # Random time stretching/compression
        rate = 1.0 + random.uniform(-max_speed, max_speed)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Ensure constant length by padding/trimming
        if len(audio) > target_sr * 10:  # If longer than 10 seconds
            audio = audio[:target_sr * 10]
        elif len(audio) < target_sr * 10:  # If shorter than 10 seconds
            padding = np.zeros(target_sr * 10 - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, padding])
    
    return audio

def stream_midi_from_s3(s3_key):
    """Stream a MIDI file directly from S3 into memory and parse it"""
    if s3_key in data_cache and 'midi' in data_cache[s3_key]:
        return data_cache[s3_key]['midi']
    
    try:
        print(f"Streaming MIDI directly from S3: {s3_key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        midi_bytes = io.BytesIO(response['Body'].read())
        midi_data = PrettyMIDI(midi_bytes)
        
        # Cache the data
        if s3_key not in data_cache:
            data_cache[s3_key] = {}
        data_cache[s3_key]['midi'] = midi_data
        
        return midi_data
    except Exception as e:
        print(f"Error streaming MIDI from S3: {s3_key}, {e}")
        return None

def stream_audio_from_s3(s3_key, target_sr=32000, apply_augmentation=False):
    """Stream audio directly from S3 into memory and convert to numpy array with optional augmentation"""
    if s3_key in data_cache and 'audio' in data_cache[s3_key]:
        audio = data_cache[s3_key]['audio']
        # Apply augmentation if needed (even for cached data)
        if apply_augmentation:
            audio = augment_audio(audio, target_sr)
        return audio
    
    try:
        print(f"Streaming audio directly from S3: {s3_key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        audio_bytes = io.BytesIO(response['Body'].read())
        
        # Use soundfile (faster than librosa for WAV files)
        audio, sr = sf.read(audio_bytes)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Apply augmentation if requested
        if apply_augmentation:
            audio = augment_audio(audio, target_sr)
        
        # Normalize audio volume
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9  # Normalize to 90% max amplitude
        
        # Ensure audio is in float32 format (not float64/double)
        audio = audio.astype(np.float32)
        
        # Cache the original (non-augmented) data
        if s3_key not in data_cache:
            data_cache[s3_key] = {}
        data_cache[s3_key]['audio'] = audio
        
        return audio
    except Exception as e:
        print(f"Error streaming audio from S3: {s3_key}, {e}")
        # Return a short array of zeros as fallback
        return np.zeros(target_sr * 3, dtype=np.float32)  # 3 seconds of silence in float32

class DirectStreamingDataset(torch.utils.data.Dataset):
    """
    Real-time Data Processing Pipeline
    
    Flow:
    1. List MIDI and Audio files from S3
    2. Match pairs using filename matching
    3. Stream files on-demand during training
    4. Apply real-time augmentation
    5. Convert to model-ready format
    
    Benefits:
    - No local storage needed
    - Dynamic data loading
    - Real-time processing
    """
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
        train_mode=True,
        test_split=0.1,
        instrument_name="flute",
        apply_augmentation=True
    ):
        self.processor = processor
        self.audio_encoder = audio_encoder
        self.device = device
        self.target_sr = target_sr
        self.max_length = int(target_sr * max_length_seconds)
        self.train_mode = train_mode
        self.instrument_name = instrument_name
        self.apply_augmentation = apply_augmentation and train_mode  # Only apply augmentation during training
        
        # List all files from S3
        print(f"Listing MIDI files from S3: {midi_dir}")
        midi_files = sorted(list_s3_files(midi_dir, extension=".midi"))
        print(f"Listing audio files from S3: {audio_dir}")
        audio_files = sorted(list_s3_files(audio_dir, extension=".wav"))
        
        print(f"Found {len(midi_files)} MIDI files and {len(audio_files)} audio files in S3")
        
        # Create paired examples using S3 paths directly
        self.examples = []
        
        # Match MIDI and audio files by name
        midi_dict = {}
        for f in midi_files:
            filename = os.path.basename(f)
            stem = os.path.splitext(filename)[0]
            
            # Try different ways to get a base name
            base_name = stem.split('_')[0]
            midi_dict[base_name] = f
            
            # Also try the full stem as a key
            midi_dict[stem] = f
        
        audio_dict = {}
        for f in audio_files:
            filename = os.path.basename(f)
            stem = os.path.splitext(filename)[0]
            
            # Try different ways to get a base name
            base_name = stem.split('_')[0]
            audio_dict[base_name] = f
            
            # Also try the full stem as a key
            audio_dict[stem] = f
        
        # Find common keys
        common_keys = set(midi_dict.keys()) & set(audio_dict.keys())
        print(f"Found {len(common_keys)} paired examples")
        
        # Create examples
        for key in common_keys:
            self.examples.append({
                "midi_path": midi_dict[key],
                "audio_path": audio_dict[key]
            })
        
        # Try fuzzy matching if needed
        if len(self.examples) < 10:
            print("Few pairs found with exact matching, trying fuzzy matching...")
            
            additional_pairs = []
            for midi_file in midi_files:
                midi_filename = os.path.basename(midi_file)
                midi_name = os.path.splitext(midi_filename)[0].lower()
                
                for audio_file in audio_files:
                    audio_filename = os.path.basename(audio_file)
                    audio_name = os.path.splitext(audio_filename)[0].lower()
                    
                    # Check for similarity
                    if (midi_name in audio_name or audio_name in midi_name or
                        any(len(substr) >= 5 for substr in set(midi_name.split('_')) & set(audio_name.split('_')))):
                        additional_pairs.append({
                            "midi_path": midi_file,
                            "audio_path": audio_file
                        })
            
            print(f"Found {len(additional_pairs)} additional pairs with fuzzy matching")
            self.examples.extend(additional_pairs)
            
            # Remove duplicates
            seen = set()
            self.examples = [x for x in self.examples if not (tuple(x.items()) in seen or seen.add(tuple(x.items())))]
        
        # Split into train and test
        if test_split > 0:
            train_examples, test_examples = train_test_split(
                self.examples, test_size=test_split, random_state=42
            )
            self.examples = train_examples if train_mode else test_examples
        
        # Limit examples if needed
        if max_examples is not None and max_examples < len(self.examples):
            self.examples = self.examples[:max_examples]
            
        print(f"Dataset contains {len(self.examples)} examples for {'training' if train_mode else 'testing'}")
        print("NOTE: All files will be streamed directly from S3 without downloading")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        midi_path = example["midi_path"]
        audio_path = example["audio_path"]
        
        # Process MIDI file directly from S3
        midi_prompt = self.process_midi(midi_path)
        
        # Process audio file directly from S3 with augmentation
        audio = stream_audio_from_s3(audio_path, self.target_sr, self.apply_augmentation)
        
        # Ensure audio length is correct
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            # Pad with zeros
            padding = np.zeros(self.max_length - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, padding])
        
        # Process prompt text
        inputs = self.processor(text=[midi_prompt], padding=True, return_tensors="pt")
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        
        # Process audio for labels
        with torch.no_grad():
            # Explicitly convert to float32
            audio_torch = torch.tensor(audio, dtype=torch.float32).to(self.device)
            # Add both batch and channel dimensions
            audio_torch = audio_torch.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, time]
            
            # Extract audio features using the model's audio encoder
            encodec_output = self.audio_encoder(audio_torch)
            
            # Get the audio codes from the EncodecOutput object
            if hasattr(encodec_output, 'audio_codes'):
                labels = encodec_output.audio_codes.squeeze(0)
                # Ensure labels are Long type for embedding layers
                if not labels.dtype == torch.long:
                    labels = labels.long()
            elif hasattr(encodec_output, 'audio_values'):
                labels = encodec_output.audio_values.squeeze(0)
                # Ensure values are converted to integer indices
                if not labels.dtype == torch.long:
                    labels = labels.long()
            else:
                # Fallback for different versions of the encoder
                try:
                    labels = encodec_output[0].squeeze(0)
                    if not labels.dtype == torch.long:
                        labels = labels.long()
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
    
    def process_midi(self, midi_path):
        """Convert MIDI to text representation by streaming from S3"""
        try:
            # Stream MIDI from S3
            midi_data = stream_midi_from_s3(midi_path)
            
            if midi_data is None:
                return f"Create a {self.instrument_name} performance based on a MIDI file. The {self.instrument_name} should play expressively."
            
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
                f"Create a {self.instrument_name} performance based on this MIDI. "
                f"The original is for {instrument_text}, with {num_notes} notes "
                f"ranging from {librosa.midi_to_note(pitch_range[0])} to "
                f"{librosa.midi_to_note(pitch_range[1])}, in {time_sig} at {tempo} BPM, "
                f"lasting {total_time} seconds. "
                f"The {self.instrument_name} should play expressively with clear articulation and good breath control."
            )
            
            return prompt
                
        except Exception as e:
            print(f"Error processing MIDI {midi_path}: {e}")
            return f"Create a {self.instrument_name} performance based on a MIDI file. The {self.instrument_name} should play expressively."


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


def direct_stream_train(
    teacher_model_path="/home/cc/Symphoniq/src/models/cached__medium_models",
    student_model_name="facebook/musicgen-small",
    midi_dir="uploads/flute/midi",
    audio_dir="uploads/flute/audio",
    output_dir="fine_tuned_model",
    batch_size=1,
    epochs=5,
    learning_rate=1e-5,
    weight_decay=0.01,  # Weight decay for L2 regularization
    alpha=0.5,
    temperature=1.0,
    warmup_steps=100,
    save_every=100,
    eval_every=50,
    max_examples=None,
    project_name="flute",
    gradient_accumulation_steps=4,  # Gradient accumulation for effectively larger batch sizes
    use_focal_loss=True,  # Whether to use focal loss
    use_spectral_norm=True,  # Whether to apply spectral normalization
    mixup_alpha=0.2  # Mixup parameter
):
    """
    Main Training Loop with Knowledge Distillation
    
    Algorithm Flow:
    1. Setup Models:
       - Load teacher (frozen) and student (trainable) models
       - Configure optimizers and loss functions
    
    2. Training Loop:
       a) Stream batch from S3
       b) Forward pass through student
       c) Get teacher's knowledge (hidden states)
       d) Compute combined loss:
          - Task Loss: How well student generates audio
          - Distillation Loss: How well student matches teacher
       e) Update student model
       f) Evaluate and save checkpoints
    
    Advanced Features:
    - Spectral Normalization: Stabilize training
    - Focal Loss: Handle class imbalance
    - Mixup Augmentation: Improve generalization
    - Gradient Accumulation: Handle larger effective batch sizes
    """
    # Set instrument name
    instrument_name = project_name
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device - use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Create dataset with direct S3 streaming and augmentation
    print(f"Creating streaming dataset from MIDI: {midi_dir} and Audio: {audio_dir}")
    
    train_dataset = DirectStreamingDataset(
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        processor=processor,
        audio_encoder=teacher_model.audio_encoder, 
        device=device,
        max_examples=max_examples,
        target_sr=32000,
        max_length_seconds=10,
        train_mode=True,
        instrument_name=instrument_name,
        apply_augmentation=True  # Enable augmentation for training
    )
    
    # Create validation dataset (no augmentation)
    val_dataset = DirectStreamingDataset(
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        processor=processor,
        audio_encoder=teacher_model.audio_encoder,
        device=device,
        max_examples=max_examples,
        target_sr=32000,
        max_length_seconds=10,
        train_mode=False,
        instrument_name=instrument_name,
        apply_augmentation=False  # No augmentation for validation
    )
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0  # Direct S3 access requires single thread
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0  # Direct S3 access requires single thread
    )
    
    # Disable codebook stacking for training stability
    if hasattr(student_model.config, "use_all_codebooks"):
        student_model.config.use_all_codebooks = False
        print("Disabled multi-codebook usage for training stability")
    
    # Apply spectral normalization
    if use_spectral_norm:
        print("Applying spectral normalization to model weights...")
        SpectralNormalization.apply(student_model)
    
    # Setup optimizer and scheduler with weight decay
    optimizer = torch.optim.AdamW(
        student_model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Use cosine schedule with warmup for better convergence
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create focal loss if needed
    if use_focal_loss:
        focal_loss_fn = FocalLoss(gamma=2.0)
        print("Using focal loss for better handling of class imbalance")
    
    # Training loop
    global_step = 0
    train_losses = []
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    # Save training config
    config = {
        "teacher_model": teacher_model_path,
        "student_model": student_model_name,
        "midi_dir": midi_dir,
        "audio_dir": audio_dir,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "alpha": alpha,
        "temperature": temperature,
        "warmup_steps": warmup_steps,
        "max_examples": max_examples,
        "project_name": project_name,
        "use_focal_loss": use_focal_loss,
        "use_spectral_norm": use_spectral_norm,
        "mixup_alpha": mixup_alpha
    }
    
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Starting advanced regularized training for {epochs} epochs with direct S3 streaming...")
    for epoch in range(epochs):
        student_model.train()
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for step, batch in enumerate(progress_bar):
            # Move all tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Ensure labels are Long type for embedding
            if not labels.dtype == torch.long:
                labels = labels.long()
            
            # Reshape the data so MusicGen understands it better
            if len(labels.shape) == 2:
                # If flat, reshape for 4 codebooks (MusicGen likes this format)
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
            
            # Sometimes mix things up to help learning
            apply_mixup = random.random() < 0.3 and mixup_alpha > 0
            if apply_mixup:
                # Blend different examples together
                mixed_labels, labels_a, labels_b, lam = mixup_data(labels, labels, mixup_alpha)
                labels = mixed_labels.long()  # Keep it in the right format
            
            # Forward pass - student
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
            
            # Standard student loss or focal loss
            if use_focal_loss and not apply_mixup:
                # Reshape logits and labels for focal loss
                logits = student_outputs.logits.view(-1, student_outputs.logits.size(-1))
                target = labels.view(-1)
                student_loss = focal_loss_fn(logits, target)
            elif apply_mixup:
                # Apply mixup criterion
                # For simplicity, we'll just use the standard loss here
                student_loss = student_outputs.loss
            else:
                student_loss = student_outputs.loss
            
            # Knowledge transfer: Extract what the teacher model knows
            with torch.no_grad():
                # Let teacher analyze the text input
                teacher_text_outputs = teacher_model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                # Grab teacher's understanding of the text
                teacher_hidden = teacher_text_outputs.hidden_states[-1]
            
            # Get student's understanding for comparison
            student_text_outputs = student_outputs.encoder_hidden_states
            student_hidden = student_text_outputs[-1]
            
            # Make sure student and teacher are speaking the same language
            if student_hidden.shape != teacher_hidden.shape:
                # Quick fix: project student's output to match teacher's dimension
                hidden_size = teacher_hidden.shape[-1]
                student_hidden = torch.nn.functional.linear(
                    student_hidden, 
                    torch.randn(hidden_size, student_hidden.shape[-1], device=device)
                )
            
            # How well does student match teacher's understanding?
            distill_loss = torch.nn.functional.mse_loss(student_hidden, teacher_hidden)
            
            # Combined loss: balance between task performance and matching teacher
            loss = (1 - alpha) * student_loss + alpha * distill_loss
            
            # Scale for gradient accumulation if needed
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Track statistics
            train_losses.append(loss.item() * gradient_accumulation_steps)  # Un-scale for logging
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            
            # Update only at the right steps or at the end of batch
            if ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == len(train_loader)):
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Update progress bar
                avg_loss = sum(epoch_losses[-gradient_accumulation_steps:]) / min(gradient_accumulation_steps, len(epoch_losses))
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                
                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    student_model.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    print(f"\nSaved checkpoint to {checkpoint_dir}")
                    
                # Evaluate
                if global_step % eval_every == 0:
                    val_loss = evaluate_model(student_model, val_loader, device, use_focal_loss, focal_loss_fn)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_dir = os.path.join(output_dir, "best_model")
                        os.makedirs(best_model_dir, exist_ok=True)
                        student_model.save_pretrained(best_model_dir)
                        processor.save_pretrained(best_model_dir)
                        print(f"\nNew best model with validation loss: {val_loss:.4f}")
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
        # End of epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"\nEpoch {epoch+1}/{epochs} complete. Average loss: {avg_epoch_loss:.4f}")
    
    # After all epochs are complete (note the indentation), save the final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    print(f"\nTraining complete! Saving final model to {final_model_path}")
    student_model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title(f"Training Loss for {project_name}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    
    return final_model_path

def evaluate_model(model, dataloader, device, use_focal_loss=False, focal_loss_fn=None):
    """
    Validation Process
    
    Purpose:
    1. Monitor training progress
    2. Detect overfitting
    3. Save best model checkpoints
    
    Method:
    - Run model on validation set
    - Compute loss metrics
    - Track best performance
    """
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Ensure labels are Long type
            if not labels.dtype == torch.long:
                labels = labels.long()
                
            # Reshape labels if needed
            if len(labels.shape) == 2:
                batch_size, seq_len = labels.shape
                labels = labels.reshape(batch_size, 1, seq_len).repeat(1, 4, 1)
            elif len(labels.shape) == 4 and labels.shape[1] == 1:
                batch_size, _, num_codebooks, seq_len = labels.shape
                labels = labels.reshape(batch_size, num_codebooks, seq_len)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if use_focal_loss:
                logits = outputs.logits.view(-1, outputs.logits.size(-1))
                target = labels.view(-1)
                loss = focal_loss_fn(logits, target)
            else:
                loss = outputs.loss
                
            val_losses.append(loss.item())
    
    model.train()
    return sum(val_losses) / len(val_losses) if val_losses else float('inf')

def validate_audio_files(audio_dir):
    """Add this validation step before training"""
    for audio_file in glob.glob(f"{audio_dir}/*.wav"):
        audio, sr = sf.read(audio_file)
        
        # Check if audio is very short/corrupted
        if len(audio) < sr * 2:  # Less than 2 seconds
            print(f"WARNING: {audio_file} is too short!")
            
        # Check for clipping/distortion
        if np.abs(audio).max() > 0.98:
            print(f"WARNING: {audio_file} may be clipped/distorted!")
            
        # Check frequency content (flute should have strong mid/high frequencies)
        freqs = np.abs(np.fft.rfft(audio))
        if np.mean(freqs[:len(freqs)//4]) > np.mean(freqs[len(freqs)//4:]):
            print(f"WARNING: {audio_file} has unusual frequency profile for flute!")

if __name__ == "__main__":
    """
    Command Line Interface
    
    Usage Example:
    python train_s3.py flute --teacher path/to/teacher --epochs 15
    
    Flow:
    1. Parse command line arguments
    2. Set up training configuration
    3. Initialize S3 streaming
    4. Start training pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MusicGen with direct S3 streaming (no downloads)")
    parser.add_argument("project_name", type=str, help="Name of the instrument/project (e.g., 'flute', 'violin')")
    parser.add_argument("--teacher", type=str, default="/home/cc/Symphoniq/src/models/cached__medium_models", 
                       help="Path to cached teacher model")
    parser.add_argument("--student", type=str, default="facebook/musicgen-small", help="Student model name")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Derive S3 directories from project name
    project_name = args.project_name
    midi_dir = f"uploads/{project_name}/midi"
    audio_dir = f"uploads/{project_name}/audio"
    output_dir = f"{project_name}_model"
    
    print(f"Starting training for {project_name} using direct S3 streaming (no downloads)...")
    print(f"Streaming MIDI files from S3 path: {midi_dir}")
    print(f"Streaming audio files from S3 path: {audio_dir}")
    print(f"Model outputs will be saved to: {output_dir}")
    
    # Start training with direct S3 streaming
    direct_stream_train(
        teacher_model_path=args.teacher,
        student_model_name=args.student,
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        output_dir=output_dir,
        batch_size=1,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha=0.7,
        temperature=0.8,
        warmup_steps=200,
        save_every=25,
        eval_every=25,
        max_examples=None,
        project_name=project_name
    )