#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate audio files in a directory and report if they are suitable for MusicGen
"""

import argparse
import os
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import sys

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
        "issues": [],
        "file_results": {}  # Add file-specific results dictionary
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
                results["file_results"][str(audio_path)] = {
                    "valid": False,
                    "issues": issues
                }
            else:
                results["valid"] += 1
                results["file_results"][str(audio_path)] = {
                    "valid": True,
                    "issues": []
                }
                
        except Exception as e:
            results["issues"].append((str(audio_path), [f"Error: {str(e)}"]))
            results["file_results"][str(audio_path)] = {
                "valid": False,
                "issues": [f"Error: {str(e)}"]
            }
    
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
    
    # Print detailed file results
    print("\nIndividual file results:")
    for file_path, result in results["file_results"].items():
        status = "✅ VALID" if result["valid"] else "❌ INVALID"
        file_name = Path(file_path).name
        print(f"{status}: {file_name}")
        if not result["valid"]:
            for issue in result["issues"]:
                print(f"  - {issue}")
    
    # Print overall recommendation
    if results["valid"] == results["total"]:
        print("\n✅ All files are valid for MusicGen training.")
    else:
        print(f"\n⚠️ {results['total'] - results['valid']} out of {results['total']} files have issues that may affect training.")
        print("Consider fixing these issues before proceeding with training.")
    
    return results

def main():
    """Main function to run audio validation"""
    parser = argparse.ArgumentParser(description='Validate audio files for MusicGen training')
    parser.add_argument('path', type=str, help='Path to directory containing audio files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output for each file')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.path):
        print(f"Error: Directory '{args.path}' does not exist.")
        sys.exit(1)
        
    # Validate audio files
    results = validate_audio_files(args.path)
    
    # If verbose mode, print detailed results for each file
    if args.verbose:
        print("\nDetailed file information:")
        for file_path, result in results["file_results"].items():
            file_name = Path(file_path).name
            print(f"\nFile: {file_name}")
            print(f"Valid: {'Yes' if result['valid'] else 'No'}")
            if result["issues"]:
                print("Issues:")
                for issue in result["issues"]:
                    print(f"  - {issue}")
            else:
                print("Issues: None")
    
    # Return overall success/failure
    return results["valid"] == results["total"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)