import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.instrumental_generator import InstrumentalGenerator

def generate_music_from_text(prompt: str, output_path: Path = None) -> str:
    """
    Generate music from text description
    Args:
        prompt (str): Text description of desired music
        output_path (Path, optional): Path to save generated audio
    Returns:
        str: Path to generated audio file
    """
    # Initialize generator
    generator = InstrumentalGenerator()
    
    # Set default output path if none provided
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "processed" / "instruments" / "generated_music.wav"
    
    # Generate music
    print(f"Generating music for prompt: {prompt}")
    result = generator.generate_instrumental(prompt, output_path)
    
    return result

if __name__ == "__main__":
    # Test generation
    test_prompt = "A smooth jazz piece with piano and saxophone, relaxing mood"
    output_file = generate_music_from_text(test_prompt)
    print(f"Generated music saved to: {output_file}")