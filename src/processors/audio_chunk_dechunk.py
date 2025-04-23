from pydub import AudioSegment
import os

def split_audio_into_chunks(file_path, chunk_length_sec=10):
    try:
        if not os.path.isfile(file_path):
            return False, f"File not found: {file_path}"

        # Load audio using from_file() to handle multiple formats robustly
        audio = AudioSegment.from_file(file_path)
        total_length_ms = len(audio)

        if total_length_ms == 0:
            return False, "Audio file has zero duration or failed to load."

        # Convert seconds to milliseconds
        chunk_length_ms = chunk_length_sec * 1000

        output_dir = os.path.join(os.path.dirname(file_path), "output_chunks")
        os.makedirs(output_dir, exist_ok=True)

        num_chunks = 0
        for i in range(0, total_length_ms, chunk_length_ms):
            chunk = audio[i:i+chunk_length_ms]
            if len(chunk) == 0:
                continue
            chunk_path = os.path.join(output_dir, f"chunk_{num_chunks + 1}.wav")
            chunk.export(chunk_path, format="wav")
            num_chunks += 1

        return True, num_chunks

    except Exception as e:
        return False, f"Error: {str(e)}"

def merge_audio_from_paths(chunk_paths, output_file="merged_output.wav"):
    try:
        if not chunk_paths:
            return False, "Chunk path list is empty."

        merged_audio = AudioSegment.empty()

        for chunk_path in chunk_paths:
            chunk_audio = AudioSegment.from_file(chunk_path)
            merged_audio += chunk_audio
            print(f"Merged: {chunk_path}")

        merged_audio.export(output_file, format="wav")
        return True, f"Merged into: {output_file}"

    except Exception as e:
        return False, f"Error: {str(e)}"
