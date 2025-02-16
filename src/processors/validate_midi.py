import pretty_midi
from pathlib import Path
from typing import Tuple, Optional

def validate_midi_file(midi_path: Path, max_duration: int = 300) -> Tuple[bool, Optional[str]]:
    """
    Validates a MIDI file for processing
    Args:
        midi_path (Path): Path to MIDI file
        max_duration (int): Maximum allowed duration in seconds (default: 300)
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        if not midi_path.exists():
            return False, f"File not found: {midi_path}"
            
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Check if file has any notes
        total_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)
        if total_notes == 0:
            return False, "MIDI file contains no notes"
            
        # Check duration but don't fail on long files
        duration = midi_data.get_end_time()
        if duration < 1:
            return False, "MIDI file too short (less than 1 second)"
            
        # Check for valid instruments
        valid_instruments = False
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                valid_instruments = True
                break
        if not valid_instruments:
            return False, "No melodic instruments found in MIDI file"
            
        return True, None
        
    except Exception as e:
        return False, f"Invalid MIDI file: {str(e)}"