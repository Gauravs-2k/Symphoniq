from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os

# Set where to save the model
cache_dir = os.path.join(os.getcwd(), "cached__medium_models")
os.makedirs(cache_dir, exist_ok=True)

print(f"Downloading MusicGen model to {cache_dir}...")

# Download model components
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

# Save to local directory
processor.save_pretrained(cache_dir)
model.save_pretrained(cache_dir)

print("Model downloaded successfully!")
print("You can now run the instrumental_generator.py script")