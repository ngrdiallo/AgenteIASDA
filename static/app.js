// ============================================================
// AGENTEAI v5 - FRONTEND CLIENT
// WebSocket communication + DOM management
// ============================================================

let ws = null;
let eventSource = null;  // For log streaming
let currentModalita = 'generale';
let currentBackend = '';  // Empty string = Auto (fallback)
let isLoading = false;
let currentFile = null;  // Store uploaded file
let lastUserMessage = '';  // Track last user message for button escalation
let lastSentFile = null;   // Track last sent file reference for escalation

const elements = {
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.querySelector('.sidebar-toggle'),
    mainContainer: document.querySelector('.main-container'),
    newChatBtn: document.querySelector('.new-chat-btn'),
    modalitaSelector: document.querySelector('.modalita-selector'),
    backendSelector: document.querySelector('.backend-selector'),
    chatArea: document.getElementById('chat-area'),
    messageInput: document.getElementById('message-input'),
    fileInput: document.getElementById('file-input'),
    sendBtn: document.getElementById('send-btn'),
    chatTitle: document.querySelector('.chat-title-box h2'),
    chatSubtitle: document.querySelector('.chat-title-box p'),
    modalitaLabel: document.querySelector('.modalita-label'),
    responseStatus: document.querySelector('.response-status'),
    modalitaInfo: document.querySelector('.modalita-info'),
    logsPanel: document.getElementById('logs-panel'),
    logsContent: document.getElementById('logs-content'),
    logsToggleBtn: document.getElementById('logs-toggle-btn'),
    logsClearBtn: document.getElementById('logs-clear-btn'),
    logsCloseBtn: document.getElementById('logs-close-btn'),
};

let modalitaData = {};
let backendData = {};

// ============================================================
// INITIALIZATION
// ============================================================

async function init() {
    connectWebSocket();
    connectLogStream();  // Start log streaming
    loadModalita();
    loadBackends();
    setupEventListeners();
    setupQuickActions();
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/chat`;
    
    ws = new WebSocket(url);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateStatus('Connesso', '#10b981');
    };
    
    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            console.log('[WS-RAW] Received message type:', message.type);
            handleWebSocketMessage(message);
        } catch (e) {
            console.error('[WS-RAW] Failed to parse message:', e, 'Raw:', event.data);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Errore connessione', '#ef4444');
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateStatus('Disconnesso', '#f97316');
        setTimeout(connectWebSocket, 3000);
    };
}

function connectLogStream() {
    /**
     * Connect to Server-Sent Events (SSE) endpoint for real-time log streaming
     * Displays logs in a bottom panel with color-coded severity levels
     */
    try {
        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
        const url = `${protocol}//${window.location.host}/api/logs/stream`;
        
        eventSource = new EventSource(url);
        
        eventSource.onmessage = (event) => {
            try {
                const logEntry = JSON.parse(event.data);
                addLogEntry(logEntry);
            } catch (e) {
                console.error('[LOG-STREAM] Failed to parse log:', e);
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('[LOG-STREAM] SSE Connection error:', error);
            // Attempt to reconnect after delay
            setTimeout(connectLogStream, 5000);
        };
        
        console.log('[LOG-STREAM] Connected to server logs');
    } catch (error) {
        console.error('[LOG-STREAM] Failed to connect:', error);
    }
}

function addLogEntry(logEntry) {
    /**
     * Add a log entry to the logs panel
     * Format: [HH:MM:SS] LEVEL | logger_name: message
     */
    if (!elements.logsPanel || !elements.logsContent) return;
    
    const timestamp = new Date(logEntry.timestamp * 1000);
    const timeStr = timestamp.toLocaleTimeString('it-IT', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
    });
    
    const logLevelClass = logEntry.level.toUpperCase();
    const logDiv = document.createElement('div');
    logDiv.classList.add('log-entry', logLevelClass);
    
    const timestampSpan = document.createElement('span');
    timestampSpan.classList.add('log-timestamp');
    timestampSpan.textContent = timeStr;
    
    const levelSpan = document.createElement('span');
    levelSpan.classList.add('log-level');
    levelSpan.textContent = logEntry.level.substring(0, 3).toUpperCase();
    
    const messageSpan = document.createElement('span');
    messageSpan.classList.add('log-message');
    messageSpan.textContent = logEntry.message;
    
    logDiv.appendChild(timestampSpan);
    logDiv.appendChild(levelSpan);
    logDiv.appendChild(messageSpan);
    
    elements.logsContent.appendChild(logDiv);
    
    // Keep only last 1000 entries to avoid memory issues
    const entries = elements.logsContent.querySelectorAll('.log-entry');
    if (entries.length > 1000) {
        entries[0].remove();
    }
    
    // Auto-scroll to bottom
    elements.logsContent.scrollTop = elements.logsContent.scrollHeight;
}

function toggleLogsPanel() {
    /**
     * Toggle visibility of the logs panel
     */
    if (elements.logsPanel) {
        elements.logsPanel.classList.toggle('hidden');
    }
}

function clearLogs() {
    /**
     * Clear all logs from the panel
     */
    if (elements.logsContent) {
        elements.logsContent.innerHTML = '';
    }
}


async function loadModalita() {
    try {
        const response = await fetch('/api/modalita');
        const data = await response.json();
        modalitaData = data;
        
        // Popola selector
        elements.modalitaSelector.innerHTML = '';
        Object.keys(data).forEach(key => {
            const option = document.createElement('option');
            option.value = key;
            const config = data[key];
            option.textContent = `${config.nome_ia} (${config.backend_preferito.split(' ')[0]})`;
            elements.modalitaSelector.appendChild(option);
        });
        
        // Set first option as selected
        elements.modalitaSelector.value = 'generale';
        updateModalitaInfo('generale');
    } catch (error) {
        console.error('Failed to load modalita:', error);
    }
}

async function loadBackends() {
    try {
        const response = await fetch('/api/backends');
        const data = await response.json();
        
        if (data.backends) {
            backendData = data.backends;
            
            // Mappa di nomi user-friendly per i backend
            const backendLabels = {
                'Groq llama-3.3-70b': '⚡ Groq (Velocissimo, gratuito)',
                'Mistral API': '🌐 Mistral Cloud (Veloce, gratuito)',
                'HuggingFace Inference': '🤗 HuggingFace (Affidabile, gratuito)',
                'Ollama Cloud': '☁️ Ollama Cloud (Gratuito in cloud)',
                'Ollama Llama2 (local)': '💻 Locale (Nel tuo PC, offline)'
            };
            
            elements.backendSelector.innerHTML = '<option value="">🤖 Automatico (Consigliato - sceglie il migliore)</option>';
            Object.entries(data.backends).forEach(([key, name]) => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = backendLabels[name] || name; // Fallback al nome tecnico se non è nella mappa
                option.title = `Provider: ${name}`; // Tooltip con nome tecnico
                elements.backendSelector.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Failed to load backends:', error);
    }
}

function setupEventListeners() {
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    elements.modalitaSelector.addEventListener('change', (e) => {
        currentModalita = e.target.value;
        updateModalitaInfo(currentModalita);
    });
    
    elements.backendSelector.addEventListener('change', (e) => {
        currentBackend = e.target.value;
        console.log('Backend selected:', currentBackend || 'Auto (Fallback)');
    });
    
    elements.newChatBtn.addEventListener('click', newChat);
    
    elements.sidebarToggle.addEventListener('click', toggleSidebar);
    
    elements.fileInput.addEventListener('change', handleFileUpload);
    
    // Log panel event listeners
    if (elements.logsToggleBtn) {
        elements.logsToggleBtn.addEventListener('click', toggleLogsPanel);
    }
    if (elements.logsCloseBtn) {
        elements.logsCloseBtn.addEventListener('click', () => {
            if (elements.logsPanel) {
                elements.logsPanel.classList.add('hidden');
            }
        });
    }
    if (elements.logsClearBtn) {
        elements.logsClearBtn.addEventListener('click', clearLogs);
    }
}

// ============================================================
// MESSAGE HANDLING
// ============================================================

function sendMessage() {
    const text = elements.messageInput.value.trim();
    
    // Allow sending if: (has text) OR (has file)
    if ((!text && !currentFile) || isLoading) return;
    
    // If file but no text, use default message
    const messageToDisplay = text || (currentFile ? `📎 ${currentFile.name}` : '');
    
    // Store last user message for button escalation
    lastUserMessage = text || `Analizza file: ${currentFile?.name || ''}`;
    
    // Display user message immediately
    addMessage(messageToDisplay, 'user');
    elements.messageInput.value = '';
    
    // Show file status if attached
    if (currentFile) {
        const preview = document.getElementById('file-preview');
        if (preview) {
            preview.innerHTML = `<span>📎 ${currentFile.name} (${(currentFile.size / 1024).toFixed(2)} KB) - Invio...</span>`;
        }
    }
    
    isLoading = true;
    elements.sendBtn.disabled = true;
    
    // Show thinking indicator
    showThinking();
    
    // Build payload - use default message if file but no text
    const payloadMessage = text || (currentFile ? `Analizza file: ${currentFile.name}` : '');
    
    const payload = {
        message: payloadMessage,
        modalita: currentModalita
    };
    
    // Add forced backend if selected
    if (currentBackend) {
        payload.forced_backend = currentBackend;
    }
    
    // Add file if present
    if (currentFile) {
        console.log('[PAYLOAD-DEBUG] currentFile exists:', JSON.stringify({
            name: currentFile.name,
            size: currentFile.size,
            file_id: currentFile.file_id,
            base64: currentFile.base64 ? 'EXISTS (' + currentFile.base64.length + ' chars)' : 'NONE',
            is_uploaded: currentFile.is_uploaded
        }));
        
        if (currentFile.file_id) {
            // Large file: Use file_id (uploaded via REST)
            payload.file_id = currentFile.file_id;
            payload.file_name = currentFile.name;
            console.log('[PAYLOAD-DEBUG] Added file_id to payload:', currentFile.file_id);
        } else {
            // Small file: Use inline base64
            payload.file_base64 = currentFile.base64;
            payload.file_name = currentFile.name;
            payload.file_size = currentFile.size;
            console.log('[PAYLOAD-DEBUG] Added inline base64 to payload (size:', currentFile.base64?.length, ')');
        }
    } else {
        console.log('[PAYLOAD-DEBUG] NO currentFile - sending text-only message');
    }
    
    // Keep last sent file reference for manual escalation
    lastSentFile = currentFile ? { ...currentFile } : null;

    console.log('[PAYLOAD-DEBUG] Final payload:', JSON.stringify(payload, null, 2));
    
    // Send via WebSocket
    ws.send(JSON.stringify(payload));
}

function handleWebSocketMessage(message) {
    console.log('[WS]', message.type.toUpperCase(), message.type === 'response' || message.type === 'error' ? message.content?.substring(0, 50) : '');
    
    if (message.type === 'thinking_stream') {
        renderThinkingStream(message);
    } else if ([
        'routing','extracting','analyzing','handoff','enriching','composing','quality_check','decision'
    ].includes(message.type)) {
        renderPhaseCard(message);
    } else if (message.type === 'attempt') {
        // Show backend attempt in the accumulated list
        const attemptsList = document.getElementById('attempts-list');
        if (attemptsList) {
            // Show the list if hidden
            attemptsList.style.display = 'block';
            
            // Add attempt as a new item (not overwrite)
            const attemptItem = document.createElement('div');
            attemptItem.className = 'attempt-item';
            attemptItem.innerHTML = `<span>🔗</span><span>${message.content}</span>`;
            attemptsList.appendChild(attemptItem);
            
            // Auto-scroll to show latest attempt
            scrollToBottom();
        }
    } else if (message.type === 'thinking') {
        updateThinking();
    } else if (message.type === 'response') {
        console.log('[RESPONSE] Received:', message.content?.substring(0, 50) + '...');
        
        // Remove thinking stream box when response arrives
        const tsBox = document.getElementById('thinking-stream-box');
        if (tsBox) tsBox.remove();
        
        // Ensure assistant message row exists
        let lastMessage = document.querySelector('.message-row.assistant:last-child .message-bubble');
        console.log('[RESPONSE] First querySelector result:', lastMessage ? 'FOUND' : 'NOT FOUND');
        
        if (!lastMessage) {
            console.log('[RESPONSE] Creating new assistant message...');
            // Create new message row if it doesn't exist
            addMessage('', 'assistant');
            lastMessage = document.querySelector('.message-row.assistant:last-child .message-bubble');
            console.log('[RESPONSE] Second querySelector result:', lastMessage ? 'FOUND' : 'NOT FOUND');
        }
        
        // Add response content to message
        if (lastMessage) {
            console.log('[RESPONSE] Setting textContent to message...');
            lastMessage.textContent = message.content;
            console.log('[RESPONSE] Message content set successfully');
        } else {
            console.error('[RESPONSE] CRITICAL: Could not find or create message bubble!');
            console.log('[RESPONSE] Current DOM state:');
            console.log(document.querySelector('.chat-area')?.innerHTML);
        }
        
        // Show metadata with quality metrics
        if (message.backend && message.latency) {
            const meta = document.querySelector('.message-row.assistant:last-child .message-meta');
            if (meta) {
                let metaHTML = `<span>${message.backend} • ${message.latency}</span>`;
                
                if (message.quality) {
                    const conf = (message.quality.confidence || 0) * 100;
                    const confColor = conf >= 70 ? '#10b981' : conf >= 50 ? '#f59e0b' : '#ef4444';
                    const depth = (message.quality.depth_score || 0).toFixed(2);
                    const words = message.quality.word_count || 0;
                    
                    metaHTML += `<span style="margin-left: 12px; color: ${confColor};"> conf ${conf.toFixed(0)}% | depth ${depth} | ${words} words</span>`;
                }
                
                if (lastSentFile) {
                    metaHTML += ` <span style="margin-left: 8px;">| 📎 ${lastSentFile.name}</span>`;
                }
                
                meta.innerHTML = metaHTML;
                console.log('[RESPONSE] Metadata set:', metaHTML);
            }
        }
        
        // Add deepen button if quality is low or can escalate
        if (message.can_escalate || (message.quality && message.quality.confidence < 0.75)) {
            const msgRow = document.querySelector('.message-row.assistant:last-child');
            if (msgRow && !msgRow.querySelector('.btn-deepen')) {
                const deepenBtn = document.createElement('button');
                deepenBtn.className = 'btn-deepen';
                deepenBtn.textContent = 'Approfondisci';
                deepenBtn.onclick = function() {
                    deepenResponse(message.content);
                };
                const meta = msgRow.querySelector('.message-meta');
                if (meta) {
                    meta.appendChild(document.createElement('br'));
                    meta.appendChild(deepenBtn);
                }
            }
        }
        
        // Clear file after response (both UI and data)
        clearFilePreview();
        clearFileData();
        
        // Scroll to show the new response
        scrollToBottom();
        
        isLoading = false;
        elements.sendBtn.disabled = false;
        console.log('[RESPONSE] Removing thinking indicator...');
        removeThinking();
        console.log('[RESPONSE] Complete');
    } else if (message.type === 'error') {
        addMessage(`Errore: ${message.content}`, 'system');
        clearFilePreview();
        isLoading = false;
        elements.sendBtn.disabled = false;
        removeThinking();
    }
}

function addMessage(text, role) {
    removeEmptyState();
    
    const row = document.createElement('div');
    row.className = `message-row ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = `message-bubble bubble-${role === 'assistant' ? 'assistant' : role === 'user' ? 'user' : 'system'}`;
    bubble.textContent = text;
    
    row.appendChild(bubble);
    
    if (role === 'assistant') {
        const meta = document.createElement('div');
        meta.className = 'message-meta';
        row.appendChild(meta);
    }
    
    elements.chatArea.appendChild(row);
    scrollToBottom();
}

function renderThinkingStream(data) {
    let thinkingBox = document.getElementById('thinking-stream-box');
    
    if (!thinkingBox) {
        thinkingBox = document.createElement('div');
        thinkingBox.id = 'thinking-stream-box';
        thinkingBox.className = 'thinking-stream-container';
        const chatContainer = document.querySelector('#chat-messages, .chat-container, .chat-area');
        if (chatContainer) chatContainer.appendChild(thinkingBox);
    }

    const phaseIcons = {
        "parsing":              "ANALISI",
        "routing":              "ROUTING",
        "first_response":       "RISPOSTA 1",
        "quality_check":        "QUALITA",
        "decision":             "DECISIONE",
        "escalation_start":     "ESCALATION",
        "escalation_prompt":    "PROMPT",
        "escalation_complete":  "COMPLETATO"
    };

    const icon = phaseIcons[data.phase] || "STEP";
    const line = document.createElement('div');
    line.className = 'thinking-stream-line';
    line.innerHTML = 
        "<span class='ts-phase'>[" + icon + "]</span> " +
        "<span class='ts-elapsed'>+" + data.elapsed_s + "s</span> " +
        "<span class='ts-content'>" + data.content + "</span>";
    thinkingBox.appendChild(line);
    thinkingBox.scrollTop = thinkingBox.scrollHeight;
}

function renderPhaseCard(data) {
    const icons = {
        routing: { emoji: '🧭', name: 'Routing' },
        extracting: { emoji: '🔍', name: 'Extractor' },
        analyzing: { emoji: '🧠', name: 'Analyzer' },
        handoff: { emoji: '🤝', name: 'Handoff' },
        enriching: { emoji: '🌐', name: 'Enricher' },
        composing: { emoji: '✍️', name: 'Explainer' },
        quality_check: { emoji: '✅', name: 'Quality Check' },
        decision: { emoji: '⚖️', name: 'Decisione' }
    };

    const config = icons[data.type] || { emoji: '⚙️', name: 'Step' };
    const modelTag = data.model ? ` <span style="color: #6b7280; font-size: 0.85em;\">(${data.model})</span>` : '';
    
    const card = document.createElement('div');
    card.className = 'phase-card';
    
    let bodyHTML = data.content || '';
    if (data.data && Object.keys(data.data).length > 0) {
        bodyHTML += '<div style="margin-top: 6px; font-size: 0.85em; color: #666;">';
        for (const [key, val] of Object.entries(data.data)) {
            if (Array.isArray(val) && val.length > 0) {
                bodyHTML += `${key}: ${val.slice(0, 3).join(', ')}${val.length > 3 ? '...' : ''}<br>`;
            }
        }
        bodyHTML += '</div>';
    }
    
    card.innerHTML = `
        <div class="phase-card-header">${config.emoji} ${config.name}${modelTag}</div>
        <div class="phase-card-body">${bodyHTML}</div>
    `;

    elements.chatArea.appendChild(card);
    scrollToBottom();
}

function showThinking() {
    removeEmptyState();
    
    const row = document.createElement('div');
    row.id = 'thinking-indicator';
    row.className = 'message-thinking';
    
    // Create thinking header
    const header = document.createElement('div');
    header.className = 'thinking-header';
    header.innerHTML = '<span>💭</span><span>Sto pensando...</span>';
    row.appendChild(header);
    
    // Create container for attempts list
    const attemptsList = document.createElement('div');
    attemptsList.id = 'attempts-list';
    attemptsList.className = 'attempts-list';
    attemptsList.style.display = 'none'; // Hidden initially
    row.appendChild(attemptsList);
    
    elements.chatArea.appendChild(row);
    scrollToBottom();
}

function updateThinking() {
    const thinking = document.getElementById('thinking-indicator');
    if (thinking) {
        thinking.style.opacity = '0.8';
    }
}

function removeThinking() {
    const thinking = document.getElementById('thinking-indicator');
    if (thinking) {
        // Collapse thinking indicator smoothly
        thinking.style.transition = 'all 0.3s ease-out';
        thinking.style.maxHeight = '0';
        thinking.style.opacity = '0';
        thinking.style.overflow = 'hidden';
        thinking.style.padding = '0';
        thinking.style.margin = '0';
        
        // Remove after animation completes
        setTimeout(() => {
            thinking.remove();
        }, 300);
    }
}

function removeEmptyState() {
    const empty = document.querySelector('.empty-state');
    if (empty) {
        empty.remove();
    }
}

function scrollToBottom() {
    // Use requestAnimationFrame for more reliable scroll timing
    // Multiple frames ensure layout is complete before scrolling
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            if (elements.chatArea) {
                const scrollHeight = elements.chatArea.scrollHeight;
                const currentTop = elements.chatArea.scrollTop;
                const containerHeight = elements.chatArea.clientHeight;
                
                // Smooth scroll to bottom
                elements.chatArea.scrollTo({
                    top: scrollHeight,
                    behavior: 'smooth'
                });
                
                console.log('[SCROLL] scrollHeight:', scrollHeight, 'top:', currentTop, 'container:', containerHeight);
            }
        });
    });
}

// ============================================================
// DEEPENING / MANUAL ESCALATION
// ============================================================

function deepenResponse(previousResponse) {
    if (!lastUserMessage) {
        addMessage('Errore: messaggio precedente non trovato', 'system');
        return;
    }
    
    isLoading = true;
    elements.sendBtn.disabled = true;
    
    showThinking();
    
    // Build request payload for /api/deepen
    const payload = {
        message: lastUserMessage,
        modalita: currentModalita,
        previous_response: previousResponse
    };
    
    // Add file reference if available
    if (lastSentFile) {
        if (lastSentFile.file_id) {
            payload.file_id = lastSentFile.file_id;
            payload.file_name = lastSentFile.name;
        } else if (lastSentFile.base64) {
            payload.file_base64 = lastSentFile.base64;
            payload.file_name = lastSentFile.name;
            payload.file_size = lastSentFile.size;
        }
    }
    
    // Send to /api/deepen endpoint
    fetch('/api/deepen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(resp => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    })
    .then(data => {
        removeThinking();
        
        // Show deepened response
        const deepResponse = `**Approfondimento:**\n\n${data.content}`;
        addMessage(deepResponse, 'assistant');
        
        // Show metadata
        const meta = document.querySelector('.message-row.assistant:last-child .message-meta');
        if (meta) {
            meta.innerHTML = `<span>${data.backend || 'Mistral'} • ${data.latency || '?'} | 🔍 Approfondimento</span>`;
        }
        
        isLoading = false;
        elements.sendBtn.disabled = false;
        scrollToBottom();
    })
    .catch(error => {
        removeThinking();
        console.error('[DEEPEN] Error:', error);
        addMessage(`Errore nell'approfondimento: ${error.message}`, 'system');
        isLoading = false;
        elements.sendBtn.disabled = false;
    });
}

// ============================================================
// UI UPDATES
// ============================================================

function updateModalitaInfo(modalita) {
    const config = modalitaData[modalita];
    if (!config) return;
    
    elements.chatTitle.textContent = config.nome_ia;
    elements.chatSubtitle.textContent = config.titolo;
    elements.modalitaLabel.textContent = config.nome_ia;
    
    const backend = config.backend_preferito.split(' ')[0] || 'Auto';
    const specialita = config.specialita ? config.specialita.join(', ') : 'Generale';
    
    elements.modalitaInfo.innerHTML = `
        <strong>${config.nome_ia}</strong><br>
        ${config.titolo}<br><br>
        <div style="font-size: 11px; color: #6b7280;">
            <div><strong>Backend:</strong> ${config.backend_preferito}</div>
            <div style="margin-top: 8px;"><strong>Specialità:</strong> ${specialita}</div>
        </div>
    `;
}

function updateStatus(text, color) {
    elements.responseStatus.textContent = text;
    elements.responseStatus.style.color = color;
}

function newChat() {
    elements.chatArea.innerHTML = `
        <div class="empty-state">
            <h2>Ciao! 👋</h2>
            <p>Sono pronto ad aiutarti. Seleziona una modalità e fammi una domanda...</p>
        </div>
    `;
}

// ============================================================
// FILE HANDLING
// ============================================================

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const fileSizeMB = file.size / (1024 * 1024);
    const fileSizeKB = (file.size / 1024).toFixed(2);
    
    // Threshold: 5MB
    const LARGE_FILE_THRESHOLD_MB = 5;
    
    if (fileSizeMB > LARGE_FILE_THRESHOLD_MB) {
        // LARGE FILE: Use REST multipart upload
        console.log(`🔄 Large file detected (${fileSizeMB.toFixed(2)} MB). Using REST upload...`);
        uploadLargeFile(file);
    } else {
        // SMALL FILE: Use inline base64 (as before)
        console.log(`✓ Small file (${fileSizeKB} KB). Using inline base64...`);
        handleSmallFileUpload(file);
    }
}

function handleSmallFileUpload(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const fileSize = (file.size / 1024).toFixed(2);
        const base64 = e.target.result;
        
        // Store file data for sending with next message
        currentFile = {
            name: file.name,
            size: file.size,
            base64: base64.split(',')[1]  // Remove 'data:application/pdf;base64,' prefix
        };
        
        // Show file preview in UI
        showFilePreview(file.name, fileSize, 'inline');
        console.log(`File loaded: ${file.name} (${fileSize} KB) - Ready to send with next message`);
    };
    
    reader.readAsDataURL(file);
}

function uploadLargeFile(file) {
    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
    
    // Show uploading status
    const preview = document.createElement('div');
    preview.id = 'file-preview';
    preview.style.cssText = `
        padding: 8px 12px;
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        font-size: 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    `;
    preview.innerHTML = `
        <span>⏳ Uploading: ${file.name} (${fileSizeMB} MB)...</span>
    `;
    
    const inputContainer = document.querySelector('.input-footer input[type="file"]').parentElement;
    const existingPreview = document.getElementById('file-preview');
    if (existingPreview) existingPreview.remove();
    inputContainer.parentElement.insertBefore(preview, inputContainer.nextSibling);
    
    // Create FormData for multipart upload
    const formData = new FormData();
    formData.append('file', file);
    
    // Send via fetch
    fetch('/api/upload-file', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    })
    .then(data => {
        if (!data.file_id) throw new Error('No file_id returned');
        
        // Store file reference
        currentFile = {
            name: data.file_name,
            size: data.file_size,
            file_id: data.file_id,  // Instead of base64
            is_uploaded: true
        };
        
        // Show success preview
        showFilePreview(data.file_name, data.file_size_mb, 'uploaded', data.file_id);
        console.log(`✓ Large file uploaded successfully: ${data.file_id}`);
    })
    .catch(error => {
        console.error('❌ Upload failed:', error);
        
        // Show error
        const errorDiv = document.getElementById('file-preview');
        if (errorDiv) {
            errorDiv.style.background = '#fee2e2';
            errorDiv.style.borderColor = '#dc2626';
            errorDiv.innerHTML = `<span>❌ Upload failed: ${error.message}</span>`;
        }
        
        clearFileData();
        setTimeout(() => clearFilePreview(), 5000);
    });
}


function showFilePreview(fileName, fileSize, type = 'inline', fileId = null) {
    // Remove existing preview
    clearFilePreview();
    
    const preview = document.createElement('div');
    preview.id = 'file-preview';
    
    // Determine styling based on file type
    let bgColor = '#dbeafe';
    let borderColor = '#1e40af';
    let icon = '📎';
    let statusText = '';
    
    if (type === 'uploaded') {
        bgColor = '#d1fae5';  // Green for uploaded
        borderColor = '#059669';
        icon = '✓';
        statusText = ' (Cloud uploaded)';
    }
    
    preview.style.cssText = `
        padding: 8px 12px;
        background: ${bgColor};
        border: 1px solid ${borderColor};
        border-radius: 8px;
        font-size: 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    `;
    
    const fileSizeDisplay = typeof fileSize === 'number' 
        ? `${fileSize.toFixed(2)} MB` 
        : `${fileSize} KB`;
    
    preview.innerHTML = `
        <span>${icon} ${fileName} (${fileSizeDisplay})${statusText}</span>
        <button onclick="clearFilePreview()" style="background: none; border: none; cursor: pointer; color: inherit; font-weight: bold;">
            ✕
        </button>
    `;
    
    const container = elements.messageInput.parentElement;
    container.insertBefore(preview, elements.messageInput);
}


function clearFilePreview() {
    const preview = document.getElementById('file-preview');
    if (preview) {
        preview.remove();
    }
    // DO NOT clear currentFile here - that should be explicit when message is sent
    // Only clear the file input UI
    elements.fileInput.value = '';  // Reset file input
}

function clearFileData() {
    // Explicitly clear currentFile (called after message is sent)
    currentFile = null;
}

// ============================================================
// SIDEBAR TOGGLE (Mobile)
// ============================================================

function toggleSidebar() {
    elements.sidebar.classList.toggle('open');
    elements.mainContainer.classList.toggle('sidebar-open');
}

// Close sidebar when clicking on main area
elements.mainContainer.addEventListener('click', () => {
    if (window.innerWidth <= 768) {
        elements.sidebar.classList.remove('open');
        elements.mainContainer.classList.remove('sidebar-open');
    }
});

// ============================================================
// NEW FEATURES: IMAGE ANALYSIS & GENERATION
// ============================================================

// Modal Management
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.style.display = 'flex';
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.style.display = 'none';
}

// ============================================================
// QUICK ACTIONS - ULTRA SIMPLE
// ============================================================

function setupQuickActions() {
    // Image Analysis quick action
    const analysisBtn = document.getElementById('quick-image-analysis-btn');
    if (analysisBtn) {
        analysisBtn.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = (e) => {
                const file = e.target.files[0];
                if (file) analyzeImageFile(file);
            };
            input.click();
        });
    }

    // Image Generation quick action
    const generationBtn = document.getElementById('quick-image-generation-btn');
    if (generationBtn) {
        generationBtn.addEventListener('click', () => {
            const prompt = prompt('Descrivi l\'immagine che vuoi generare:\n(Es: A beautiful Italian landscape with mountains and sunset)');
            if (prompt) generateImage(prompt);
        });
    }
}

// Analyze uploaded image
async function analyzeImageFile(file) {
    const reader = new FileReader();
    reader.onload = async (event) => {
        const base64 = event.target.result.split(',')[1];
        
        // Add loading message to chat
        addMessage('📸 Sto analizzando l\'immagine...', 'system');
        
        try {
            const response = await fetch('/api/image-analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: 'Analizzami questa immagine in modo dettagliato',
                    modalita: 'analisi',
                    file_base64: base64,
                    file_name: file.name
                })
            });

            const data = await response.json();
            if (data.success) {
                addMessage(data.content, 'assistant');
            } else {
                addMessage('❌ Errore nell\'analisi: ' + (data.error || 'Errore sconosciuto'), 'system');
            }
        } catch (error) {
            addMessage('❌ Errore: ' + error.message, 'system');
        }
    };
    reader.readAsDataURL(file);
}

// Generate image with Flux 2 Max
async function generateImage(prompt) {
    addMessage('🎨 Sto generando l\'immagine...', 'system');
    
    try {
        const response = await fetch('/api/generate-art', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: prompt,
                modalita: 'creative'
            })
        });

        const data = await response.json();
        if (data.success) {
            // Display image in chat
            const messageRow = document.createElement('div');
            messageRow.className = 'message-row assistant';
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble bubble-assistant';
            bubble.innerHTML = `
                <div>
                    <strong>🎨 Generata da ${data.generator}</strong><br>
                    <img src="${data.image_url}" style="max-width: 100%; max-height: 400px; border-radius: 8px; margin-top: 8px;">
                    <p style="font-size: 0.85em; color: #666; margin-top: 8px;">${prompt}</p>
                </div>
            `;
            
            messageRow.appendChild(bubble);
            elements.chatArea.appendChild(messageRow);
            elements.chatArea.scrollTop = elements.chatArea.scrollHeight;
        } else {
            addMessage('❌ Errore nella generazione: ' + (data.error || 'Errore sconosciuto'), 'system');
        }
    } catch (error) {
        addMessage('❌ Errore: ' + error.message, 'system');
    }
}

// ============================================================
// STARTUP
// ============================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
