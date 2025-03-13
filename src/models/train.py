#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune MusicGen for instrumental performance using DataLoader
"""

import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    get_linear_schedule_with_warmup
)

# Fix the imports - separate dataset imports from data_processing imports
from data_processing import validate_audio_files
from dataset import AudioMIDIDataset, custom_collate_fn

def train_musicgen(
    model_name="facebook/musicgen-small",
    midi_dir="/Users/gauravs/Documents/Symphoniq/src/data/training/midi",
    audio_dir="/Users/gauravs/Documents/Symphoniq/src/data/training/audio",
    output_dir="fine_tuned_model",
    instrument="flute",
    batch_size=1,
    epochs=5,
    learning_rate=1e-5,
    warmup_steps=100,
    save_every=100,
    eval_every=50,
    max_examples=None
):
    """
    Train MusicGen model on instrument dataset
    
    Args:
        model_name: Hugging Face model name
        midi_dir: Directory containing MIDI files
        audio_dir: Directory containing audio files
        output_dir: Directory to save the model
        instrument: Target instrument name
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps for scheduler
        save_every: Save checkpoint every N steps
        eval_every: Evaluate every N steps
        max_examples: Maximum number of examples to use
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device - use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Fix the decoder_start_token_id issue
    config = MusicgenForConditionalGeneration.config_class.from_pretrained(model_name)
    config.decoder_start_token_id = 0
    model = MusicgenForConditionalGeneration.from_pretrained(model_name, config=config)
    model.to(device)
    
    # Create dataset
    print(f"Creating dataset from {audio_dir}")
    train_dataset = AudioMIDIDataset(
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        instrument=instrument,
        processor=processor,
        audio_encoder=model.audio_encoder,
        device=device,
        max_examples=max_examples,
        target_sr=32000,
        max_length_seconds=10
    )
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0  # Increase if you have more CPU cores
    )
    
    # Disable codebook stacking for training stability
    if hasattr(model.config, "use_all_codebooks"):
        model.config.use_all_codebooks = False
        print("Disabled multi-codebook usage for training stability")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    global_step = 0
    train_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
                # In the training loop, modify this part:
        for batch in progress_bar:
            # Move all tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Fix labels shape for codebook compatibility
            labels = batch["labels"].to(device)
            batch_size, seq_len = labels.shape
            
            # Reshape labels for MusicGen's codebook expectation (4 codebooks)
            labels = labels.reshape(batch_size, 1, seq_len).repeat(1, 4, 1)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update progress
            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())
            
            # Save checkpoint
            global_step += 1
            if global_step % save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
            
            # Evaluate
            if global_step % eval_every == 0:
                # Generate a sample for qualitative evaluation
                model.eval()
                with torch.no_grad():
                    # Use first example from batch for generation
                    prompt = train_dataset[0]["prompt"]
                    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                    
                    # Generate audio
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        guidance_scale=3.0,
                        temperature=0.7
                    ).cpu().numpy()
                
                # Save generated audio
                audio_path = output_dir / f"sample_{global_step}.wav"
                try:
                    import scipy.io.wavfile
                    scipy.io.wavfile.write(
                        audio_path, 
                        rate=32000, 
                        data=(audio_values[0] * 32767).astype(np.int16)
                    )
                    print(f"Saved sample audio to {audio_path}")
                except Exception as e:
                    print(f"Error saving audio sample: {str(e)}")
                
                # Switch back to train mode
                model.train()
        
        # End of epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs} complete. Average loss: {avg_loss:.4f}")
    
    # Save final model
    final_model_path = output_dir / "final_model"
    model.save_pretrained(final_model_path)
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
    # Fix the paths for running as main script
    import sys
    
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
    
    # Validate audio files
    audio_dir = Path("/Users/gauravs/Documents/Symphoniq/src/data/training/audio")
    validation_results = validate_audio_files(audio_dir)
    
    if validation_results["valid"] > 0:
        # Start training
        train_musicgen()
    else:
        print("Please fix audio file issues before training")