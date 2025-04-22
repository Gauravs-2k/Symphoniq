import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to the path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import for feature extraction and quantization
from feature_extraction import extract_features_from_file, VocalFeatureExtractor
from quantization import quantize_features
from dict_map import map_features_to_prompt

def print_array_info(name, array):
    """Print useful information about numpy arrays without overwhelming output."""
    print(f"{name}:")
    print(f"  Shape: {array.shape}")
    print(f"  Data type: {array.dtype}")
    print(f"  Range: [{np.min(array):.4f}, {np.max(array):.4f}]")
    print(f"  Mean: {np.mean(array):.4f}")

def plot_quantized_features(quantized, save_path=None):
    """Plot the quantized feature representations"""
    plt.figure(figsize=(12, 10))
    
    # Plot MFCC token sequence
    plt.subplot(3, 1, 1)
    plt.title("MFCC Token Sequence")
    plt.plot(quantized['mfcc_tokens'], '-o', markersize=4)
    plt.xlabel("Frame")
    plt.ylabel("Token ID")
    
    # Plot melody contour from pitch tokens
    plt.subplot(3, 1, 2)
    plt.title("Pitch Contour")
    # Filter out negative values (unvoiced frames)
    pitch_tokens = np.array(quantized['pitch_tokens'])
    valid_pitch = pitch_tokens > 0
    plt.plot(np.arange(len(pitch_tokens))[valid_pitch], 
             pitch_tokens[valid_pitch], '-o', markersize=4)
    plt.xlabel("Frame")
    plt.ylabel("MIDI Note")
    
    # Plot rhythm token sequence
    plt.subplot(3, 1, 3)
    plt.title("Rhythm Tokens (Beat Strengths)")
    plt.stem(quantized['rhythm_tokens'])
    plt.xlabel("Beat")
    plt.ylabel("Strength (0-4)")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main(audio_file_path, instrument="flute", visualize=False, save_plot=None):
    """Extract features from an audio file and print the results."""
    # Validate the file path
    if not os.path.exists(audio_file_path):
        print(f"Error: File {audio_file_path} not found!")
        return 1
    
    print(f"Extracting features from: {audio_file_path}")
    
    # Extract features
    features = extract_features_from_file(audio_file_path)
    
    # Print basic audio information
    print("\nAudio Information:")
    print(f"  Sample rate: {features['sample_rate']} Hz")
    # Calculate duration here
    duration = len(features['audio_data']) / features['sample_rate']
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(features['audio_data'])}")
    
    # Print information about extracted features
    print("\nExtracted Features:")
    print_array_info("Spectrogram", features['spectrogram'])
    print_array_info("Mel Spectrogram", features['mel_spectrogram'])
    print_array_info("MFCCs", features['mfcc'])
    
    # Now quantize the features and display histograms
    print("\nQuantizing features to token sequences...")
    
    # Create a specific save directory for analysis files using a relative path
    # Go up three levels (prompt -> models -> src -> Symphoniq) then down to data/input/separated/analysis
    analysis_dir = Path(__file__).parent.parent.parent / "data/input/separated/analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create a full path for the quantization histogram in the analysis directory
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    quantization_hist_path = os.path.join(analysis_dir, f"{base_name}_quantization_hist.png")
    
    print(f"Will save quantization histograms to: {quantization_hist_path}")
    
    # Call quantize_features with explicit save_path
    try:
        # Always generate the plot, show it only if visualize is True
        quantized = quantize_features(features, plot=visualize, save_path=quantization_hist_path)
    except Exception as e:
        print(f"Error during quantization plotting: {e}")
        # If plotting fails, try without plotting
        quantized = quantize_features(features, plot=False)
    
    # Print quantized feature information
    print("\nQuantized Features:")
    print(f"  MFCC tokens: {len(quantized['mfcc_tokens'])} tokens")
    print(f"  Mel tokens: {len(quantized['mel_tokens'])} tokens")
    print(f"  Pitch tokens: {len(quantized['pitch_tokens'])} tokens")
    print(f"  Rhythm tokens: {len(quantized['rhythm_tokens'])} tokens")
    
    # Print example tokens
    print("\nExample Token Sequences:")
    print(f"  MFCC tokens (first 10): {quantized['mfcc_tokens'][:10]}")
    print(f"  Mel tokens (first 10): {quantized['mel_tokens'][:10]}")
    print(f"  Pitch tokens (first 10): {quantized['pitch_tokens'][:10]}")
    print(f"  Rhythm tokens (all): {quantized['rhythm_tokens']}")
    
    # Map features to a single MusicGen prompt, passing duration
    print("\nGenerating single MusicGen prompt from features...")
    # Pass duration to the mapping function
    prompt_to_save = map_features_to_prompt(quantized, instrument, duration) 
    
    print(f"\nAudio duration: {duration:.2f} seconds")
    
    # Save the single generated prompt to file
    prompt_file = os.path.splitext(audio_file_path)[0] + "_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt_to_save)
    
    # Estimate token count for the single prompt
    token_count = len(prompt_to_save.split()) * 1.3 
    print(f"\nSaved final MusicGen prompt to: {prompt_file}")
    print(f"  Prompt: \"{prompt_to_save}\"")
    print(f"  Estimated token count: ~{token_count:.0f} (target < 512)") 
    
    # Adjust warnings for token count (applied to the single prompt)
    if duration <= 10 and token_count < 200: 
        print(f"  Warning: Token count ({token_count:.0f}) is low for a short clip.")
        print("           Consider enhancing the prompt generation logic in dict_map.py for more detail.")
    
    # Visualize if requested
    if visualize:
        print("\nGenerating visualization...")
        # Original features
        extractor = VocalFeatureExtractor()
        extractor.plot_features(features, save_path=save_plot)
        
        # Quantized features
        quantized_plot = save_plot.replace('.png', '_quantized.png') if save_plot else None
        plot_quantized_features(quantized, save_path=quantized_plot)
    
    print("\nFeature extraction and prompt generation completed successfully!")
    # Update return dictionary to only include the single prompt
    return {
        "features": features, 
        "quantized": quantized,
        "prompt": prompt_to_save # Changed from "prompts" dictionary
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features and generate MusicGen prompts")
    parser.add_argument("audio_file", help="Path to the audio WAV file")
    parser.add_argument("-i", "--instrument", default="guitar", 
                       help="Target instrument for the prompt (default: flute)")
    parser.add_argument("-v", "--visualize", action="store_true", 
                        help="Visualize the extracted features")
    parser.add_argument("-s", "--save", metavar="PATH",
                        help="Save the visualization to the specified path")
    
    args = parser.parse_args()
    
    # Call main() to extract features and exit
    results = main(args.audio_file, args.instrument, args.visualize, args.save)
    # Example of accessing the prompt: print(results['prompt'])