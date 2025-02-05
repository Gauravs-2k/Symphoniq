# Symphoniq

AI-powered audio processing pipeline for vocal separation and instrumental generation.

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd Symphoniq

# Create virtual environment
python -m venv venv

# Activate virtual environment (Mac/Linux) 
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Project Structure

symphoniq/
├── src/
│   ├── converters/    # Audio format conversion
│   ├── processors/    # Audio processing modules
│   ├── models/        # ML models
│   └── utils/         # Helper functions
├── data/
│   ├── input/mp3      # Source MP3 files
│   ├── processed/     # Intermediate files
│   └── output/        # Final output files


## Processing Pipeline
1. Processing Pipeline
2. Convert MP3 to WAV
3. Separate vocals from instruments
4. Convert vocals to MIDI
5. Generate new instrumental
6. Merge audio tracks
7. Convert final WAV to MP3

## Usage
1. Place MP3 files in data/input/mp3/
2. Run 

python src/main.py

Core Dependencies
pydub (audio conversion)
spleeter (vocal separation)
pretty_midi (MIDI conversion)
tensorflow (instrumental generation)
soundfile (WAV processing)