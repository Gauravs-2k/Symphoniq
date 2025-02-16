from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pathlib import Path

def download_model(model_name="facebook/musicgen-small"):
    cache_dir = Path(__file__).parent / "cached_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    
    print(f"Saving model to {cache_dir}...")
    processor.save_pretrained(cache_dir)
    model.save_pretrained(cache_dir)
    print("Model downloaded and saved successfully!")

if __name__ == "__main__":
    download_model()