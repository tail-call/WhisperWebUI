const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileInfo    = document.getElementById('fileInfo');
const fileName    = document.getElementById('fileName');
const fileSize    = document.getElementById('fileSize');
const status      = document.getElementById('status');
const statusText  = document.getElementById('statusText');
const spinner     = document.getElementById('spinner');
const playerSec   = document.getElementById('playerSection');
const audioEl     = document.getElementById('audioEl');
const subtitleBox = document.getElementById('subtitleBox');
const result      = document.getElementById('result');
const resultText  = document.getElementById('resultText');
const chunksSec   = document.getElementById('chunksSection');
const chunksBody  = document.getElementById('chunksBody');
const resetBtn    = document.getElementById('resetBtn');

let selectedFile  = null;
let audioBlobURL  = null;   // Blob URL for client-side playback
let subtitleLines = [];     // [{start, end, text, el}]

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function fmtTime(s) {
  if (s == null) return '—';
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return m > 0 ? `${m}:${sec.padStart(4, '0')}` : `${sec}s`;
}

function showStatus(msg, type) {
  statusText.textContent = msg;
  spinner.style.display  = type === 'loading' ? 'block' : 'none';
  status.className       = 'status visible ' + type;
}
function hideStatus() { status.className = 'status'; }

/* ---- Drag & Drop ---- */
['dragenter', 'dragover'].forEach(e =>
  dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); })
);
['dragleave', 'drop'].forEach(e =>
  dropZone.addEventListener(e, () => dropZone.classList.remove('drag-over'))
);

dropZone.addEventListener('drop', ev => {
  ev.preventDefault();
  if (ev.dataTransfer.files.length) selectFile(ev.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) selectFile(fileInput.files[0]);
});

function selectFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = fmtBytes(file.size);
  fileInfo.classList.add('visible');
  playerSec.classList.remove('visible');
  result.classList.remove('visible');
  chunksSec.classList.remove('visible');
  resetBtn.style.display = 'none';
  submit();
}

/* ---- Upload ---- */
async function submit() {
  if (!selectedFile) return;

  const apiBase = document.location.origin;
  const formData = new FormData();
  formData.append('file', selectedFile);

  showStatus('Transcribing…', 'loading');

  try {
    const res = await fetch(apiBase + '/transcribe', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `Server error ${res.status}`);
    }
    const data = await res.json();
    showStatus('Done!', 'success');

    resultText.textContent = data.text || '(empty transcript)';
    result.classList.add('visible');

    if (data.chunks && data.chunks.length) {
      chunksBody.innerHTML = data.chunks.map(c =>
        `<tr><td>${fmtTime(c.timestamp[0])}</td><td>${fmtTime(c.timestamp[1])}</td><td>${escapeHtml(c.text)}</td></tr>`
      ).join('');
      chunksSec.classList.add('visible');
    }

    setupPlayer(data.chunks);

    resetBtn.style.display = 'inline-block';
  } catch (err) {
    showStatus(err.message, 'error');
  }
}

/* ---- Player & Subtitles ---- */
function setupPlayer(chunks) {
  // Revoke previous Blob URL to free memory
  if (audioBlobURL) { URL.revokeObjectURL(audioBlobURL); }

  // Create a Blob URL so the audio stays in the client
  const blob = new Blob([selectedFile], { type: selectedFile.type || 'audio/wav' });
  audioBlobURL = URL.createObjectURL(blob);
  audioEl.src = audioBlobURL;
  subtitleLines = [];
  subtitleBox.innerHTML = '';

  if (!chunks || !chunks.length) {
    playerSec.classList.add('visible');
    return;
  }

  // Build subtitle cue elements from chunks
  subtitleLines = chunks.map(c => {
    const line = document.createElement('div');
    line.className = 'sub-line';
    line.innerHTML = `<span class="sub-time">${fmtTime(c.timestamp[0])}</span>${escapeHtml(c.text)}`;
    line.addEventListener('click', () => {
      audioEl.currentTime = c.timestamp[0];
      if (audioEl.paused) audioEl.play();
    });
    subtitleBox.appendChild(line);
    return { start: c.timestamp[0], end: c.timestamp[1], el: line };
  });

  playerSec.classList.add('visible');

  // Sync subtitles on audio time update
  audioEl.addEventListener('timeupdate', onAudioTimeUpdate);

  // When audio ends, clear active subtitle
  audioEl.addEventListener('ended', () => {
    subtitleLines.forEach(s => s.el.classList.remove('active'));
  });
}

function onAudioTimeUpdate() {
  const t = audioEl.currentTime;
  let activeIdx = -1;

  for (let i = subtitleLines.length - 1; i >= 0; i--) {
    if (t >= subtitleLines[i].start) { activeIdx = i; break; }
  }

  subtitleLines.forEach((s, i) => {
    s.el.classList.toggle('active', i === activeIdx);
  });

  // Auto-scroll active subtitle into view
  if (activeIdx >= 0) {
    const activeEl = subtitleLines[activeIdx].el;
    const boxRect = subtitleBox.getBoundingClientRect();
    const lineRect = activeEl.getBoundingClientRect();
    const offset = lineRect.top - boxRect.top - subtitleBox.clientHeight / 2 + lineRect.height / 2;
    subtitleBox.scrollTop += offset;
  }
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function copyText() {
  navigator.clipboard.writeText(resultText.textContent);
  const btn = document.getElementById('copyBtn');
  btn.textContent = '✅ Copied!';
  setTimeout(() => btn.textContent = '📋 Copy', 1500);
}

function reset() {
  // Revoke Blob URL to free client memory
  if (audioBlobURL) { URL.revokeObjectURL(audioBlobURL); audioBlobURL = null; }

  selectedFile = null;
  fileInput.value = '';
  subtitleLines = [];
  audioEl.src = '';
  subtitleBox.innerHTML = '';
  playerSec.classList.remove('visible');
  fileInfo.classList.remove('visible');
  result.classList.remove('visible');
  chunksSec.classList.remove('visible');
  resetBtn.style.display = 'none';
  hideStatus();
}
