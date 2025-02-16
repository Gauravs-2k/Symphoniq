from pydub import AudioSegment
import os

def convert_to_mp3(input_wav, output_mp3=None):
    """Convert WAV to MP3"""
    try:
        if output_mp3 is None:
            output_mp3 = os.path.splitext(input_wav)[0] + '.mp3'
            
        audio = AudioSegment.from_wav(input_wav)
        audio.export(output_mp3, format='mp3')
        print(f"Successfully converted to MP3: {output_mp3}")
        return output_mp3
    except Exception as e:
        print(f"Error converting to MP3: {str(e)}")
        return None