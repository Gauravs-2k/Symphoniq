from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from main import process_audio
import tempfile

app = Flask(__name__, static_folder='ui')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'input')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow only MP3 files
ALLOWED_EXTENSIONS = {'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('ui', path)

@app.route('/api', methods=['POST'])
def convert_audio():
    # Check if file exists in request
    if 'mp3_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['mp3_file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is an MP3
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only MP3 files are allowed'}), 400
    
    # Get the instrument parameter
    instrument = request.form.get('instrument', 'guitar')
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Process the audio file
        result_dir = process_audio(file_path, instrument)
        
        if not result_dir:
            return jsonify({'error': 'Processing failed'}), 500
        
        # Get the final MP3 path
        base_name = os.path.splitext(filename)[0]
        final_mp3 = os.path.join(result_dir, f"{base_name}_final.mp3")
        
        # Create relative paths for preview and download URLs
        relative_path = os.path.relpath(final_mp3, os.path.dirname(__file__))
        download_url = f"/download/{os.path.basename(final_mp3)}"
        preview_url = download_url  # Using the same URL for preview and download
        
        return jsonify({
            'success': True,
            'previewUrl': preview_url,
            'downloadUrl': download_url,
            'filename': os.path.basename(final_mp3)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    final_dir = os.path.join(os.path.dirname(__file__), 'data', 'input', 'final')
    return send_from_directory(final_dir, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
