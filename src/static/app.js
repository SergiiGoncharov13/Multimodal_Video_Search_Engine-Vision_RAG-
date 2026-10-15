// DOM Elements
const chat = document.getElementById("chat-messages");
const player = document.getElementById("player");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const cueReadout = document.getElementById("cue-readout");
const videoLabel = document.getElementById("active-video-label");
const wsDot = document.getElementById("ws-dot");
const wsStatus = document.getElementById("ws-status");
const pillsContainer = document.getElementById("video-pills-container");
const errorOverlay = document.getElementById("video-error-overlay");
const overlayTitle = document.getElementById("overlay-error-title");
const overlayDesc = document.getElementById("overlay-error-desc");
const overlayActionBtn = document.getElementById("overlay-action-btn");
const fileInput = document.getElementById("videoUpload");
const fileChosenLabel = document.getElementById("fileChosenLabel");
const uploadStatus = document.getElementById("uploadStatus");
const uploadBtn = document.getElementById("uploadBtn");

let activeAiMsg = null;
let lastCues = [];
let diskVideos = [];
let dbVideos = [];
let currentVideoName = "";
let ws = null;
let ingestPollTimer = null;

// File Input Display
function handleFileSelected() {
    if (fileInput.files.length > 0) {
        fileChosenLabel.textContent = fileInput.files[0].name;
        uploadStatus.textContent = "Ready to upload";
        uploadStatus.className = "upload-status";
    } else {
        fileChosenLabel.textContent = "Choose MP4 Video";
    }
}

// Fetch available videos from server
async function loadVideoCatalog() {
    try {
        const resp = await fetch("/api/videos");
        if (resp.ok) {
            const data = await resp.json();
            diskVideos = data.disk_videos || [];
            dbVideos = data.db_videos || [];
            renderVideoPills();

            // If no video is selected yet, default to the first available video on disk
            if (!currentVideoName && diskVideos.length > 0) {
                selectVideo(diskVideos[0]);
            }
        }
    } catch (err) {
        console.warn("Failed to fetch video catalog:", err);
    }
}

function renderVideoPills() {
    pillsContainer.innerHTML = "";
    if (diskVideos.length === 0) {
        pillsContainer.innerHTML = `<span style="font-size:0.76rem; color:var(--danger)">No MP4 videos in data/videos/</span>`;
        return;
    }

    diskVideos.forEach(v => {
        const pill = document.createElement("button");
        pill.className = `video-pill ${v === currentVideoName ? 'active' : ''}`;
        pill.innerHTML = `<span>▶</span> ${v}`;
        pill.onclick = () => selectVideo(v);
        pillsContainer.appendChild(pill);
    });
}

function selectVideo(videoName) {
    currentVideoName = videoName;
    videoLabel.textContent = videoName;
    renderVideoPills();
    hideVideoError();

    const targetUrl = `/videos/${encodeURIComponent(videoName)}`;
    const currentPath = player.currentSrc ? new URL(player.currentSrc, window.location.href).pathname : '';
    
    if (currentPath !== targetUrl) {
        player.src = targetUrl;
        player.load();
        cueReadout.textContent = `Loaded ${videoName}`;
    }
}

function showVideoError(videoName, message) {
    overlayTitle.textContent = `Video File Not Found: "${videoName}"`;
    overlayDesc.innerHTML = message;
    if (diskVideos.length > 0 && diskVideos[0] !== videoName) {
        overlayActionBtn.textContent = `Switch to ${diskVideos[0]}`;
        overlayActionBtn.style.display = "inline-block";
        overlayActionBtn.onclick = () => selectVideo(diskVideos[0]);
    } else {
        overlayActionBtn.style.display = "none";
    }
    errorOverlay.style.display = "flex";
    cueReadout.innerHTML = `<span style="color:var(--danger)">⚠️ ${videoName} missing on disk</span>`;
}

function hideVideoError() {
    errorOverlay.style.display = "none";
}

function recoverVideoPlayback() {
    if (diskVideos.length > 0) {
        selectVideo(diskVideos[0]);
    }
}

player.onerror = () => {
    const requestedName = currentVideoName || "requested video";
    showVideoError(requestedName, `The video file <strong>${requestedName}</strong> could not be loaded from <code>data/videos/</code>.`);
};

// Safe HTML5 Seeking with loadedmetadata listener
function seekTo(videoName, seconds) {
    if (diskVideos.length > 0 && !diskVideos.includes(videoName)) {
        showVideoError(videoName, `Video <strong>${videoName}</strong> is referenced in the search database, but the physical file is not in <code>data/videos/</code>.`);
        return;
    }

    hideVideoError();

    const applySeek = () => {
        player.currentTime = Math.max(0, seconds);
        player.play().catch(e => console.log("Autoplay paused by browser:", e));
        cueReadout.innerHTML = `Playing at <strong>${formatTime(seconds)}</strong> in <strong>${videoName}</strong>`;
    };

    const targetUrl = `/videos/${encodeURIComponent(videoName)}`;
    const currentPath = player.currentSrc ? new URL(player.currentSrc, window.location.href).pathname : '';

    if (currentPath !== targetUrl) {
        currentVideoName = videoName;
        videoLabel.textContent = videoName;
        renderVideoPills();
        player.src = targetUrl;
        
        const onLoaded = () => {
            player.removeEventListener("loadedmetadata", onLoaded);
            applySeek();
        };
        player.addEventListener("loadedmetadata", onLoaded);
        player.load();
    } else {
        applySeek();
    }
}

function stepVideo(delta) {
    if (player && !isNaN(player.duration)) {
        player.currentTime = Math.max(0, Math.min(player.duration, player.currentTime + delta));
    }
}

function formatTime(sec) {
    const s = Math.floor(sec);
    const mins = Math.floor(s / 60);
    const remainingSecs = s % 60;
    return `${mins}:${remainingSecs.toString().padStart(2, '0')}`;
}

// Markdown Renderer
function renderMarkdown(rawText) {
    if (typeof marked !== "undefined" && marked.parse) {
        return marked.parse(rawText);
    }
    return rawText
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

// Clean agent response text from Python artifacts
function cleanAgentText(raw) {
    let text = raw.replace(/^(\u{1F9E0}\s*)?Final [Aa]nswer:\s*/u, "");
    
    if (text.startsWith("[{'type': 'text'") || text.startsWith('[{"type": "text"')) {
        try {
            const matches = [...text.matchAll(/'text':\s*'([^']*)'/g)];
            if (matches.length > 0) {
                text = matches.map(m => m[1]).join("\n");
            }
        } catch(e) {}
    }
    return text;
}

// Video Upload & Ingestion Tracking
async function uploadVideo() {
    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please select an MP4 video file first.");
        return;
    }
    
    const file = fileInput.files[0];
    if (!file.name.toLowerCase().endsWith(".mp4")) {
        alert("Only .mp4 video files are supported.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    
    uploadStatus.textContent = "Uploading video...";
    uploadStatus.className = "upload-status processing";
    uploadBtn.disabled = true;
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            uploadStatus.textContent = "✅ Uploaded! Ingestion started in background...";
            uploadStatus.className = "upload-status success";
            
            appendSystemMessage(`⚙️ System: Uploaded '${file.name}'. Extracting keyframes and generating vision embeddings...`);
            fileInput.value = "";
            fileChosenLabel.textContent = "Choose MP4 Video";
            
            // Reload available videos catalog immediately
            await loadVideoCatalog();
            
            // Start polling ingestion status
            pollIngestStatus(file.name);
        } else {
            uploadStatus.textContent = `❌ ${result.detail || 'Upload failed'}`;
            uploadStatus.className = "upload-status error";
        }
    } catch (error) {
        console.error("Upload failed:", error);
        uploadStatus.textContent = "❌ Connection failed during upload.";
        uploadStatus.className = "upload-status error";
    } finally {
        uploadBtn.disabled = false;
    }
}

function pollIngestStatus(videoName) {
    if (ingestPollTimer) clearInterval(ingestPollTimer);

    ingestPollTimer = setInterval(async () => {
        try {
            const resp = await fetch("/api/ingest/status");
            if (resp.ok) {
                const tasks = await resp.json();
                const task = tasks[videoName];
                if (task) {
                    if (task.status === "processing") {
                        uploadStatus.textContent = `⏳ ${task.step || 'Processing...'}`;
                        uploadStatus.className = "upload-status processing";
                    } else if (task.status === "completed") {
                        uploadStatus.textContent = `🎉 Ingestion complete for ${videoName}!`;
                        uploadStatus.className = "upload-status success";
                        appendSystemMessage(`✅ Ingestion finished for '${videoName}'! You can now search for scenes in this video.`);
                        clearInterval(ingestPollTimer);
                        await loadVideoCatalog();
                        selectVideo(videoName);
                    } else if (task.status === "failed") {
                        uploadStatus.textContent = `❌ Ingestion failed: ${task.error || 'Unknown error'}`;
                        uploadStatus.className = "upload-status error";
                        clearInterval(ingestPollTimer);
                    }
                }
            }
        } catch (e) {
            console.warn("Status poll error:", e);
        }
    }, 2500);
}

// WebSocket Connection
function connectWebSocket() {
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${wsProto}//${window.location.host}/ws/ask`);

    ws.onopen = () => {
        wsDot.classList.remove("disconnected");
        wsStatus.textContent = "Agent Ready";
    };

    ws.onclose = () => {
        wsDot.classList.add("disconnected");
        wsStatus.textContent = "Reconnecting...";
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };

    ws.onmessage = (event) => {
        const data = event.data;

        // Finished Streaming
        if (data === "[DONE]") {
            removeTypingIndicator();
            if (activeAiMsg && lastCues.length > 0) {
                renderCueChips(activeAiMsg, lastCues);
            }
            activeAiMsg = null;
            lastCues = [];
            setControls(true);
            return;
        }

        // Metadata Cues Payload
        if (data.startsWith("__METADATA__")) {
            try {
                const parsed = JSON.parse(data.replace("__METADATA__", ""));
                if (parsed.type === "PLAYBACK_CUES") {
                    lastCues = parsed.cues || [];
                    
                    // Auto-play the first playable cue if video exists on disk
                    const firstPlayable = lastCues.find(c => c.file_exists !== false && (diskVideos.length === 0 || diskVideos.includes(c.video_name)));
                    if (firstPlayable) {
                        seekTo(firstPlayable.video_name, firstPlayable.timestamp_sec);
                    } else if (lastCues.length > 0 && lastCues[0].file_exists === false) {
                        cueReadout.innerHTML = `<span style="color:var(--warning)">⚠️ Best match is in "${lastCues[0].video_name}" (not in data/videos)</span>`;
                    }
                }
            } catch(e) {
                console.error("Failed to parse metadata", e);
            }
            return;
        }

        // Intermediate System updates
        if (data.startsWith("🔄") || data.startsWith("✅")) {
            removeTypingIndicator();
            appendSystemMessage(data);
            showTypingIndicator();
            activeAiMsg = null;
            return;
        }

        // AI Answer
        removeTypingIndicator();
        if (!activeAiMsg) {
            activeAiMsg = document.createElement("div");
            activeAiMsg.className = "msg msg-ai";
            activeAiMsg.innerHTML = `
                <div class="msg-header">🤖 Video Analysis Agent</div>
                <div class="markdown-content"></div>
            `;
            chat.appendChild(activeAiMsg);
        }

        const contentEl = activeAiMsg.querySelector(".markdown-content");
        const cleaned = cleanAgentText(data);
        contentEl.innerHTML = renderMarkdown(cleaned);
        chat.scrollTop = chat.scrollHeight;
    };
}

function renderCueChips(container, cues) {
    const header = document.createElement("div");
    header.className = "cue-strip-header";
    header.innerHTML = `<span>🎯 Matched Scenes & Moments (${cues.length}):</span>`;
    container.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "cue-grid";

    cues.forEach(cue => {
        const card = document.createElement("div");
        const isPlayable = cue.file_exists !== false && (diskVideos.length === 0 || diskVideos.includes(cue.video_name));
        card.className = `cue-card ${isPlayable ? '' : 'missing-file'}`;

        const thumbHtml = cue.frame_url ?
            `<img class="cue-thumb" src="${cue.frame_url}" alt="Frame thumbnail" onerror="this.outerHTML='<div class=\\'cue-thumb-fallback\\'>🎬</div>'">` :
            `<div class="cue-thumb-fallback">${cue.modality === 'audio' ? '🎙️' : '🎬'}</div>`;

        const statusBadge = isPlayable ?
            `<span class="cue-status-badge playable">● Play</span>` :
            `<span class="cue-status-badge missing">⚠️ Missing file</span>`;

        card.innerHTML = `
            ${thumbHtml}
            <div class="cue-details">
                <div class="cue-meta-row">
                    <span class="cue-time-badge">⏱️ ${cue.formatted || cue.timestamp_sec + 's'}</span>
                    <span class="cue-video-name" title="${cue.video_name}">${cue.video_name}</span>
                    <span class="cue-modality-badge">${cue.modality || 'visual'}</span>
                    ${statusBadge}
                </div>
                <div class="cue-caption" title="${cue.caption || ''}">${cue.caption || 'Scene match'}</div>
            </div>
        `;

        card.onclick = () => seekTo(cue.video_name, cue.timestamp_sec);
        grid.appendChild(card);
    });

    container.appendChild(grid);
    chat.scrollTop = chat.scrollHeight;
}

function sendQuery() {
    const q = queryInput.value.trim();
    if (!q || !ws || ws.readyState !== WebSocket.OPEN) return;

    setControls(false);
    appendUserMessage(q);
    showTypingIndicator();
    ws.send(q);
    queryInput.value = "";
}

function quickQuery(text) {
    queryInput.value = text;
    sendQuery();
}

function appendUserMessage(text) {
    const el = document.createElement("div");
    el.className = "msg msg-user";
    el.innerHTML = `
        <div class="msg-header">👤 You</div>
        <div>${escapeHtml(text)}</div>
    `;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
}

function appendSystemMessage(text) {
    const el = document.createElement("div");
    el.className = `msg-system ${text.startsWith("🔄") ? 'tool-call' : (text.startsWith("✅") ? 'tool-result' : '')}`;
    el.textContent = text;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
}

function showTypingIndicator() {
    removeTypingIndicator();
    const ind = document.createElement("div");
    ind.id = "typing-ind";
    ind.className = "msg msg-ai";
    ind.style.width = "auto";
    ind.style.display = "inline-flex";
    ind.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chat.appendChild(ind);
    chat.scrollTop = chat.scrollHeight;
}

function removeTypingIndicator() {
    const ind = document.getElementById("typing-ind");
    if (ind) ind.remove();
}

function setControls(enabled) {
    queryInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    if (enabled) queryInput.focus();
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
    loadVideoCatalog();
    connectWebSocket();
    queryInput.focus();
});
