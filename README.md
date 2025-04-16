# Symphoniq - InstruGen AI

A web application that converts vocals in MP3 files to instrumental tracks using AI.

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
```

## Setup and Installation

1. Install the required dependencies:

```bash
pip install flask numpy matplotlib
```

2. Make sure you have all the necessary model dependencies installed.

## Running the Application

1. Navigate to the src directory:

```bash
cd /Users/gauravs/Documents/Symphoniq/src
```

2. Run the Flask server:

```bash
python app.py
```

3. Open your browser and go to:

```
http://localhost:5000
```

4. Use the web interface to:
   - Upload an MP3 file
   - Select the target instrument
   - Click "Convert" to process the audio
   - After processing, you can listen to or download the generated music

## Project Structure

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

1. Convert MP3 to WAV
2. Separate vocals from instruments
3. Convert vocals to MIDI
4. Generate new instrumental
5. Merge audio tracks
6. Convert final WAV to MP3

## Usage

1. Place MP3 files in data/input/mp3/
2. Run 

```bash
python src/main.py
```

## Core Dependencies

- pydub (audio conversion)
- spleeter (vocal separation)
- pretty_midi (MIDI conversion)
- tensorflow (instrumental generation)
- soundfile (WAV processing)

## System Requirements

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari)
- Sufficient disk space for audio processing