from pydub import AudioSegment
import os
import subprocess

def check_dependencies():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install using: brew install ffmpeg")
        return False

def convert_to_wav(input_path, output_path=None):
    """Convert audio file to WAV format"""
    if not check_dependencies():
        return None
        
    try:
        # Verify input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Set output path
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '.wav'
        
        # Load and convert audio
        print(f"Converting {input_path} to WAV format...")
        audio = AudioSegment.from_file(input_path)
        
        # Set audio properties
        audio = audio.set_frame_rate(44100)
        audio = audio.set_channels(2)
        
        # Export
        audio.export(output_path, format='wav')
        print(f"Conversion complete: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "../data/input/eterna-cancao-wav-12569.mp3"
    if os.path.exists(input_file):
        convert_to_wav(input_file)
    else:
        print(f"Input file not found: {input_file}")