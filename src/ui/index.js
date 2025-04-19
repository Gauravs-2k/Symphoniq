document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("mp3_file");
  const selectedFile = document.getElementById("selected-file");
  const selectedFilename = document.getElementById("selected-filename");
  const playBtn = document.getElementById("play-original");
  const removeBtn = document.getElementById("remove-file");
  const originalAudio = document.getElementById("original-audio");
  const progressBar = document.getElementById("progress-bar");
  const timeDisplay = document.getElementById("time-display");
  const convertBtn = document.getElementById("convertBtn");
  const instrumentSelect = document.getElementById("instrument");
  const convertedAudio = document.getElementById("converted-audio");
  const convertedAudioContainer = document.getElementById("converted-audio-container");
  // const downloadBtn = document.getElementById("downloadBtn");

  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => e.preventDefault());

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragenter", () => dropZone.classList.add("drag-active"));
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-active"));

  dropZone.addEventListener("drop", (e) => {
    dropZone.classList.remove("drag-active");
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".mp3")) handleFile(file);
    else alert("Please select an MP3 file.");
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file && file.name.endsWith(".mp3")) handleFile(file);
    else alert("Please select an MP3 file.");
  });

  function handleFile(file) {
    selectedFilename.textContent = file.name;
    selectedFile.classList.remove("hidden");
    dropZone.classList.add("hidden");

    const objectUrl = URL.createObjectURL(file);
    originalAudio.src = objectUrl;
    progressBar.style.width = "0%";
    timeDisplay.textContent = "0:00 / 0:00";
    updatePlayIcon(false);
  }

  playBtn.addEventListener("click", () => {
    if (originalAudio.paused) {
      originalAudio.play();
      updatePlayIcon(true);
    } else {
      originalAudio.pause();
      updatePlayIcon(false);
    }
  });
  originalAudio.addEventListener("loadedmetadata", () => {
    timeDisplay.textContent = `0:00 / ${formatTime(originalAudio.duration)}`;
  });
  

  originalAudio.addEventListener("timeupdate", () => {
    const percent = (originalAudio.currentTime / originalAudio.duration) * 100;
    progressBar.style.width = `${percent}%`;
    timeDisplay.textContent = `${formatTime(originalAudio.currentTime)} / ${formatTime(originalAudio.duration)}`;
  });

  originalAudio.addEventListener("ended", () => {
    updatePlayIcon(false);
    progressBar.style.width = "0%";
  });

  function updatePlayIcon(playing) {
    playBtn.querySelector("svg").innerHTML = playing
      ? `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />`
      : `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
         <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />`;
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? "0" + s : s}`;
  }

  removeBtn.addEventListener("click", () => {
    fileInput.value = "";
    originalAudio.pause();
    originalAudio.src = "";
    dropZone.classList.remove("hidden");
    selectedFile.classList.add("hidden");
    convertedAudioContainer.classList.add("hidden");
    // downloadBtn.disabled = true;
  });

  convertBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file || !file.name.endsWith(".mp3")) return alert("Please select a valid MP3 file.");
    const instrument = instrumentSelect.value;

    convertBtn.disabled = true;
    convertBtn.textContent = "Converting...";

    const formData = new FormData();
    formData.append("mp3_file", file);
    formData.append("instrument", instrument);

    try {
      const res = await fetch("/api", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Failed");
      const data = await res.json();

      convertedAudio.src = data.previewUrl;
      convertedAudioContainer.classList.remove("hidden");
      // downloadBtn.disabled = false;
      // downloadBtn.onclick = () => {
      //   const a = document.createElement("a");
      //   a.href = data.downloadUrl;
      //   a.download = data.filename || "converted_song.mp3";
      //   a.click();
      // };
    } catch (err) {
      console.log(err);
      alert("Conversion failed.");
    } finally {
      convertBtn.disabled = false;
      convertBtn.textContent = "Convert";
    }
  });
});
