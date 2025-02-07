import os
import torch
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

def separate_audio(input_file, output_dir='separated'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading model...")
        model = pretrained.get_model('htdemucs')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        print("Loading audio file...")
        wav = AudioFile(input_file).read(streams=0, samplerate=model.samplerate)
        
        # Normalize audio
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        print("Separating tracks...")
        sources = apply_model(model, wav[None], device=device, progress=True)[0]
        sources = sources * ref.std() + ref.mean()
        
        print("Saving separated tracks...")
        for idx, name in enumerate(model.sources):
            output_file = f'{output_dir}/{os.path.splitext(os.path.basename(input_file))[0]}_{name}.wav'
            save_audio(sources[idx], output_file, samplerate=model.samplerate)
            print(f"Saved {name} track to {output_file}")
        
        print(f"Separation complete. Files saved in {output_dir}/")
        return True
        
    except Exception as e:
        print(f"Error during separation: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "../data/input/eterna-cancao-wav-12569.wav"
    if os.path.exists(input_file):
        separate_audio(input_file)
    else:
        print(f"Input file not found: {input_file}")