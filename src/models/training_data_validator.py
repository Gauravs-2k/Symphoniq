#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate audio files in S3 bucket for MusicGen flute training and remove bad ones
"""

import os
import io
import numpy as np
import librosa
import soundfile as sf
import boto3
from tqdm import tqdm
import argparse

# S3 Configuration - reuse from train_s3.py
S3_CONFIG = {
    'bucket': 'iit-symphoniq',
    'aws_access_key_id': 'AKIAQUFLP4IVKRLIXANY',
    'aws_secret_access_key': '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',
    'region_name': 'ap-south-1'
}

def validate_s3_audio_files(s3_prefix, delete_bad=False, dry_run=True):
    """
    Validate audio files in the S3 bucket and optionally delete bad ones
    
    Args:
        s3_prefix: S3 path prefix (e.g., "uploads/flute/audio")
        delete_bad: Whether to delete bad files
        dry_run: If True, don't actually delete files, just report
        
    Returns:
        Lists of good and bad files
    """
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=S3_CONFIG['aws_access_key_id'],
        aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
        region_name=S3_CONFIG['region_name']
    )
    bucket_name = S3_CONFIG['bucket']
    
    # List all audio files in the prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
    
    audio_files = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.wav', '.mp3')):
                    audio_files.append(key)
    
    print(f"Found {len(audio_files)} audio files in S3 bucket at {s3_prefix}")
    
    good_files = []
    bad_files = []
    issues = {}
    
    for audio_key in tqdm(audio_files, desc="Validating audio files"):
        try:
            # Stream audio directly from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=audio_key)
            audio_bytes = io.BytesIO(response['Body'].read())
            
            # Load audio
            audio, sr = sf.read(audio_bytes)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            
            issue = None
            
            # Check 1: Duration - too short?
            if len(audio) < sr * 2:  # Less than 2 seconds
                issue = "too_short"
            
            # Check 2: Clipping/distortion
            elif np.abs(audio).max() > 0.98:
                issue = "clipping"
            
            # Check 3: Frequency profile (specialized for flute)
            else:
                # Get the spectral centroid (average frequency weighted by amplitude)
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                avg_centroid = np.mean(spectral_centroid)
                
                # Flute-specific frequency bands and their expected energy distribution
                # Fundamental range (~250-2000 Hz) and harmonics (2000-5000 Hz)
                freqs = np.abs(np.fft.rfft(audio))
                freq_bins = np.linspace(0, sr/2, len(freqs))
                
                # Define flute-specific frequency ranges
                fund_low_idx = np.searchsorted(freq_bins, 250)
                fund_high_idx = np.searchsorted(freq_bins, 2000)
                harm_high_idx = np.searchsorted(freq_bins, 5000)
                
                # Calculate energy in different bands
                fund_energy = np.mean(freqs[fund_low_idx:fund_high_idx])
                harm_energy = np.mean(freqs[fund_high_idx:harm_high_idx])
                low_energy = np.mean(freqs[:fund_low_idx])
                
                # Flute profile issues:
                # 1. If average centroid is too low (below 800 Hz)
                # 2. If low freq energy is too dominant compared to fundamental range
                # 3. If harmonic energy is too low compared to fundamental
                if avg_centroid < 800:
                    issue = "spectral_centroid_too_low"
                elif low_energy > fund_energy * 1.5:  # Much more lenient
                    issue = "too_much_low_frequency"
                elif harm_energy < fund_energy * 0.2:  # Harmonics should be present
                    issue = "weak_harmonics"
            
            # Check 4: SNR (signal-to-noise ratio)
            if issue is None:
                # Estimate noise level in the quietest sections
                frame_length = 2048
                hop_length = 512
                rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
                noise_threshold = np.percentile(rms, 10)  # Estimate of noise floor
                signal_level = np.mean(rms)
                
                if signal_level > 0 and noise_threshold > 0:
                    snr = 20 * np.log10(signal_level / noise_threshold)
                    if snr < 15:  # Less than 15dB SNR
                        issue = "low_snr"
            
            if issue:
                bad_files.append(audio_key)
                issues[audio_key] = issue
                print(f"❌ {audio_key}: FAIL - {issue}")
            else:
                good_files.append(audio_key)
                print(f"✅ {audio_key}: PASS")
                
        except Exception as e:
            print(f"Error processing {audio_key}: {str(e)}")
            bad_files.append(audio_key)
            issues[audio_key] = f"error: {str(e)}"
    
    # Summary
    print("\n" + "="*50)
    print(f"VALIDATION COMPLETE: {len(good_files)} good, {len(bad_files)} bad files")
    
    if bad_files:
        print("\nBAD FILES:")
        for file in bad_files:
            print(f"  - {file}: {issues.get(file, 'unknown issue')}")
    
    # Delete bad files if requested
    if delete_bad and bad_files:
        if dry_run:
            print("\nDRY RUN: Would delete these files:")
            for file in bad_files:
                print(f"  - {file}")
        else:
            print("\nDeleting bad files from S3...")
            for file in tqdm(bad_files, desc="Deleting"):
                try:
                    s3_client.delete_object(Bucket=bucket_name, Key=file)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")
    
    return good_files, bad_files, issues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate S3 audio files for MusicGen training")
    parser.add_argument("instrument", type=str, help="Instrument name (e.g., flute)")
    parser.add_argument("--delete", action="store_true", help="Delete bad files from S3")
    parser.add_argument("--real-delete", action="store_true", help="Actually delete files (otherwise dry run)")
    
    args = parser.parse_args()
    
    # Set S3 prefix based on instrument
    s3_prefix = f"uploads/{args.instrument}/audio"
    
    # Run validation
    good_files, bad_files, issues = validate_s3_audio_files(
        s3_prefix=s3_prefix,
        delete_bad=args.delete or args.real_delete,
        dry_run=not args.real_delete
    )
    
    # Save results to local file
    with open(f"{args.instrument}_validation_results.txt", "w") as f:
        f.write(f"Validation results for {args.instrument}\n")
        f.write(f"Total files: {len(good_files) + len(bad_files)}\n")
        f.write(f"Good files: {len(good_files)}\n")
        f.write(f"Bad files: {len(bad_files)}\n\n")
        
        if bad_files:
            f.write("BAD FILES:\n")
            for file in bad_files:
                f.write(f"  - {file}: {issues.get(file, 'unknown issue')}\n")