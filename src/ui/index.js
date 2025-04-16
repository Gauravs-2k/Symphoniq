// convert.js

document.addEventListener('DOMContentLoaded', () => {
  const convertBtn = document.getElementById('convertBtn');
  const fileInput = document.getElementById('mp3_file');
  const instrumentSelect = document.getElementById('instrument');
  const previewBtn = document.getElementById('previewBtn');
  const downloadBtn = document.getElementById('downloadBtn');

  // Initially disable preview and download buttons
  previewBtn.disabled = true;
  downloadBtn.disabled = true;

  convertBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
      alert('Please select an MP3 file.');
      return;
    }

    if (!file.name.toLowerCase().endsWith('.mp3')) {
      alert('The selected file must be an MP3.');
      return;
    }

    const instrument = instrumentSelect.value;
    if (!instrument) {
      alert('Please select an instrument.');
      return;
    }

    convertBtn.disabled = true;
    convertBtn.textContent = 'Converting...';
    previewBtn.disabled = true;
    downloadBtn.disabled = true;

    const formData = new FormData();
    formData.append('mp3_file', file);
    formData.append('instrument', instrument);

    try {
      const response = await fetch('/api', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Conversion failed with status ' + response.status);
      }

      const result = await response.json();
      console.log('Conversion result:', result);

      // Enable the preview and download buttons
      previewBtn.disabled = false;
      downloadBtn.disabled = false;

      previewBtn.onclick = () => {
        window.open(result.previewUrl, '_blank');
      };

      downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = result.downloadUrl;
        a.download = result.filename || 'converted_song.mp3';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      };

      alert('Conversion completed successfully!');

    } catch (error) {
      console.error('Error during conversion:', error);
      alert('Conversion failed. Please try again.');
    } finally {
      convertBtn.disabled = false;
      convertBtn.textContent = 'Convert';
    }
  });
});
