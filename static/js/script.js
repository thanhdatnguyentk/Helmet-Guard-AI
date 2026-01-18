const CONFIG = {
    API_BASE: '/api',
    POLL_INTERVAL: 1000
};

const UI = {
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    uploadPartition: document.getElementById('uploadPartition'),
    resultPartition: document.getElementById('resultPartition'),
    resultVideo: document.getElementById('resultVideo'),
    violatorCount: document.getElementById('violatorCount'),
    helmetCount: document.getElementById('helmetCount'),
    progressBar: document.getElementById('progressBar'),
    progressContainer: document.getElementById('progressContainer'),
    statusText: document.getElementById('statusText'),
    historyList: document.getElementById('historyList'),
    downloadBtn: document.getElementById('downloadBtn'),
    violatorGallery: document.getElementById('violatorGallery'),
    galleryGrid: document.getElementById('galleryGrid'),
    
    updateProgress(percent, text) {
        this.progressBar.style.width = `${percent}%`;
        this.statusText.textContent = text;
    },
    
    resetUpload() {
        this.uploadBtn.style.display = 'block';
        this.progressContainer.style.display = 'none';
        this.statusText.style.display = 'none';
        this.fileInput.value = '';
        this.dropZone.querySelector('p').textContent = `Drag and drop or click to select video`;
    },
    
    showProcessing() {
        this.uploadBtn.style.display = 'none';
        this.progressContainer.style.display = 'block';
        this.statusText.style.display = 'block';
    }
};

let currentVideoFile = "";

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    setupEventListeners();
});

function setupEventListeners() {
    UI.dropZone.addEventListener('click', () => UI.fileInput.click());
    
    UI.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        UI.dropZone.style.borderColor = 'var(--primary)';
    });
    
    UI.dropZone.addEventListener('dragleave', () => {
        UI.dropZone.style.borderColor = 'var(--glass-border)';
    });
    
    UI.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length) {
            UI.fileInput.files = files;
            updateDropZoneText(files[0].name);
        }
    });

    UI.fileInput.addEventListener('change', () => {
        if (UI.fileInput.files.length) {
            updateDropZoneText(UI.fileInput.files[0].name);
        }
    });

    UI.uploadBtn.addEventListener('click', handleUpload);
    UI.downloadBtn.addEventListener('click', () => {
        if (currentVideoFile) window.location.href = `${CONFIG.API_BASE}/download/${currentVideoFile}`;
    });
}

function updateDropZoneText(name) {
    UI.dropZone.querySelector('p').textContent = `Selected: ${name}`;
}

async function handleUpload() {
    if (!UI.fileInput.files.length) {
        alert("Please select a video first!");
        return;
    }

    const formData = new FormData();
    formData.append('video', UI.fileInput.files[0]);

    UI.showProcessing();
    UI.updateProgress(0, "Uploading video...");

    try {
        const response = await fetch(`${CONFIG.API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.success) {
            pollStatus(data.filename);
        } else {
            alert("Upload failed: " + (data.error || "Unknown error"));
            UI.resetUpload();
        }
    } catch (error) {
        console.error(error);
        alert("An error occurred during upload.");
        UI.resetUpload();
    }
}

function pollStatus(filename) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${CONFIG.API_BASE}/status/${filename}`);
            if (!response.ok) throw new Error("Status API failed");
            
            const data = await response.json();

            if (data.status === "processing" || data.status === "starting") {
                UI.updateProgress(data.progress || 0, data.status === "starting" ? "Initializing AI..." : `Analyzing: ${data.progress}%`);
                syncGallery(data.violator_images || []);
            } else if (data.status === "completed") {
                clearInterval(interval);
                showResult(filename, data);
                loadHistory(); 
            } else if (data.status === "failed") {
                clearInterval(interval);
                alert("Processing failed: " + (data.error || "Reason unknown"));
                UI.resetUpload();
            }
        } catch (e) {
            console.error("Polling error:", e);
        }
    }, CONFIG.POLL_INTERVAL);
}

function syncGallery(images) {
    if (images.length > 0) {
        UI.violatorGallery.style.display = 'block';
        const existingCount = UI.galleryGrid.querySelectorAll('.gallery-item').length;
        if (images.length > existingCount) {
            for (let i = existingCount; i < images.length; i++) {
                const item = document.createElement('div');
                item.className = 'gallery-item';
                item.innerHTML = `<img src="/crops/${images[i]}" alt="Violator" title="Click to open" onclick="window.open(this.src)">`;
                UI.galleryGrid.appendChild(item);
            }
        }
    }
}

function showResult(filename, stats) {
    UI.uploadPartition.style.display = 'none';
    UI.resultPartition.style.display = 'flex';
    
    UI.violatorCount.textContent = stats.violators;
    UI.helmetCount.textContent = stats.helmets;

    const videoSource = UI.resultVideo.querySelector('source');
    videoSource.src = `/results/${filename}`;
    UI.resultVideo.load();
    UI.resultVideo.play();

    currentVideoFile = filename;
    UI.downloadBtn.style.display = 'flex';

    UI.galleryGrid.innerHTML = '';
    syncGallery(stats.violator_images || []);
}

async function loadHistory() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/history`);
        const history = await response.json();
        
        UI.historyList.innerHTML = '';
        if (!history || history.length === 0) {
            UI.historyList.innerHTML = '<p style="text-align:center; color:rgba(255,255,255,0.3); margin-top:20px;">No records found</p>';
            return;
        }

        history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            
            div.innerHTML = `
                <i class="fa-solid fa-circle-play" onclick="event.stopPropagation(); playFromHistory('${item.name}')"></i>
                <div class="history-info" onclick="playFromHistory('${item.name}')">
                    <span class="history-name">${item.name}</span>
                    <div class="history-meta">
                        <span><i class="fa-solid fa-user-slash"></i> ${item.violators}</span> | 
                        <span><i class="fa-solid fa-user-check"></i> ${item.helmets}</span>
                    </div>
                </div>
                <i class="fa-solid fa-trash delete-btn" onclick="event.stopPropagation(); deleteHistory('${item.name}')"></i>
            `;
            UI.historyList.appendChild(div);
        });
    } catch (e) {
        logger.error("Failed to load history", e);
    }
}

async function playFromHistory(filename) {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/status/${filename}`);
        const data = await response.json();
        if (data.status === "completed") {
            showResult(filename, data);
        }
    } catch (e) {
        console.error(e);
    }
}

async function deleteHistory(filename) {
    if (!confirm("Are you sure you want to delete this recorded session?")) return;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/history/${filename}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        if (data.success) {
            loadHistory();
            if (currentVideoFile === filename) {
                window.location.reload();
            }
        }
    } catch (e) {
        console.error(e);
        alert("Deletion failed!");
    }
}
