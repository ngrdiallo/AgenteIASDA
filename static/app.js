// ============================================================
// AGENTEIA - Artificial Scholar Interface
// ============================================================

class AgenteIA {
    constructor() {
        this.messages = [];
        this.currentMode = 'chat';
        this.currentBackend = 'auto';
        this.isLoading = false;
        
        // Multiple files support
        this.uploadedFiles = []; // {id, name, type, knowledge}
        
        // Current conversation context
        this.conversationContext = "";
        
        this.chatId = this.loadChatId();
        
        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.loadChatHistory();
        this.updateStatus('Pronto');
        this.updateFilesDisplay();
    }

    bindElements() {
        // Input
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.fileInput = document.getElementById('file-input');
        this.filePreview = document.getElementById('file-preview');
        this.previewImage = document.getElementById('preview-image');
        this.previewFilename = document.getElementById('preview-filename');
        this.removeFileBtn = document.getElementById('remove-file');

        // Chat
        this.messagesContainer = document.getElementById('messages');
        this.welcomeScreen = document.getElementById('welcome-screen');
        this.chatTitle = document.getElementById('chat-title');
        this.chatContainer = document.getElementById('chat-container');

        // Sidebar
        this.sidebar = document.getElementById('sidebar');
        this.menuBtn = document.getElementById('menu-btn');
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.modeBtns = document.querySelectorAll('.mode-btn');
        this.backendSelector = document.getElementById('backend-selector');
        this.quickActions = document.querySelectorAll('.quick-action');
    }

    bindEvents() {
        // Auto-grow input
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
        });

        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // File upload - allow multiple
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.removeFileBtn.addEventListener('click', () => this.removeAllFiles());

        // Sidebar
        this.menuBtn.addEventListener('click', () => this.toggleSidebar());
        this.newChatBtn.addEventListener('click', () => this.newChat());

        // Mode selection
        this.modeBtns.forEach(btn => {
            btn.addEventListener('click', () => this.setMode(btn.dataset.mode));
        });

        // Backend selection
        this.backendSelector.addEventListener('change', (e) => {
            this.currentBackend = e.target.value;
        });

        // Quick actions
        this.quickActions.forEach(btn => {
            btn.addEventListener('click', () => {
                this.messageInput.value = btn.dataset.prompt;
                this.messageInput.focus();
            });
        });

        // Save chat before leaving
        window.addEventListener('beforeunload', () => this.saveChatHistory());
    }

    // ============================================================
    // CORE FUNCTIONS
    // ============================================================

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        if (this.isLoading) return;

        this.hideWelcome();
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.setLoading(true);

        try {
            // Build context from uploaded files
            let contextFromFiles = "";
            if (this.uploadedFiles.length > 0) {
                contextFromFiles = "\n\n📄 DOCUMENTI ALLEGATI:\n";
                this.uploadedFiles.forEach((f, i) => {
                    contextFromFiles += `\n[DOCUMENTO ${i+1}: ${f.name}]\n`;
                    if (f.knowledge) {
                        if (f.knowledge.movements) contextFromFiles += `- Movimenti: ${f.knowledge.movements.join(', ')}\n`;
                        if (f.knowledge.years) contextFromFiles += `- Periodi: ${f.knowledge.years.slice(0,10).join(', ')}\n`;
                        if (f.knowledge.key_concepts) contextFromFiles += `- Concetti: ${f.knowledge.key_concepts.slice(0,10).join(', ')}\n`;
                    }
                    contextFromFiles += "\n";
                });
            }

            // Build prompt with context
            let fullPrompt = message;
            if (contextFromFiles) {
                fullPrompt = `L'utente chiede: "${message}"${contextFromFiles}

Rispondi analizzando i documenti allegati e crea collegamenti tra di essi.`;
            }

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    message: fullPrompt,
                    has_documents: this.uploadedFiles.length > 0
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            
            if (data.response) {
                // Add source info if we have documents
                let responseText = data.response;
                if (this.uploadedFiles.length > 0) {
                    responseText += `\n\n📚 Fonti: ${this.uploadedFiles.map(f => f.name).join(', ')}`;
                }
                this.addMessage(responseText, 'assistant');
            } else {
                this.addMessage('Risposta ricevuta', 'assistant');
            }

        } catch (error) {
            console.error('Error:', error);
            this.addMessage(`❌ Errore: ${error.message}`, 'assistant');
        } finally {
            this.setLoading(false);
            this.saveChatHistory();
        }
    }

    async handleFileUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;

        this.addMessage(`📄 Caricamento di ${files.length} file...`, 'system');
        this.setLoading(true);

        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Upload failed');
                }

                const data = await response.json();
                
                if (data.success) {
                    this.uploadedFiles.push({
                        id: data.file_id,
                        name: data.filename,
                        type: data.type,
                        knowledge: data.knowledge || null
                    });
                }
            }

            this.updateFilesDisplay();
            
            // Show summary
            let infoMsg = `✅ Caricati ${files.length} documento/i:\n`;
            this.uploadedFiles.forEach((f, i) => {
                infoMsg += `\n${i+1}. ${f.name}`;
                if (f.knowledge) {
                    if (f.knowledge.movements?.length) infoMsg += `\n   🎨 ${f.knowledge.movements.slice(0,3).join(', ')}`;
                    if (f.knowledge.years?.length) infoMsg += `\n   📅 ${f.knowledge.years.slice(0,3).join(', ')}`;
                }
            });
            infoMsg += '\n\nOra puoi fare domande sui documenti!';
            
            this.addMessage(infoMsg, 'system');

        } catch (error) {
            console.error('Upload error:', error);
            this.addMessage(`❌ Errore upload: ${error.message}`, 'assistant');
        } finally {
            this.setLoading(false);
            // Clear input so user can select same files again
            this.fileInput.value = '';
        }
    }

    updateFilesDisplay() {
        if (this.uploadedFiles.length > 0) {
            // Show combined preview
            const names = this.uploadedFiles.map(f => f.name).join(', ');
            this.previewFilename.textContent = `${this.uploadedFiles.length} file: ${names.substring(0, 50)}${names.length > 50 ? '...' : ''}`;
            this.filePreview.style.display = 'flex';
            this.previewImage.style.display = 'none';
        } else {
            this.filePreview.style.display = 'none';
        }
    }

    removeAllFiles() {
        this.uploadedFiles = [];
        this.updateFilesDisplay();
    }

    addMessage(content, role) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        
        const avatarIcon = role === 'user' ? 'U' : (role === 'system' ? '📄' : '🤓');
        
        messageEl.innerHTML = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-content">
                <div class="message-bubble">${this.formatContent(content)}</div>
            </div>
        `;

        this.messagesContainer.appendChild(messageEl);
        this.messages.push({ content, role });
        this.scrollToBottom();
    }

    addThinking() {
        const thinkingEl = document.createElement('div');
        thinkingEl.className = 'message assistant';
        thinkingEl.id = 'thinking-message';
        thinkingEl.innerHTML = `
            <div class="message-avatar">🤓</div>
            <div class="message-content">
                <div class="thinking">
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span>Sto analizzando e creando collegamenti...</span>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(thinkingEl);
        this.scrollToBottom();
    }

    removeThinking() {
        const thinkingEl = document.getElementById('thinking-message');
        if (thinkingEl) thinkingEl.remove();
    }

    // ============================================================
    // UI HELPERS
    // ============================================================

    formatContent(content) {
        if (!content) return '';
        
        let formatted = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    hideWelcome() {
        this.welcomeScreen.style.display = 'none';
    }

    showWelcome() {
        this.welcomeScreen.style.display = 'flex';
        this.messagesContainer.innerHTML = '';
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.sendBtn.disabled = loading;
        this.messageInput.disabled = loading;
        
        if (loading) {
            this.addThinking();
        } else {
            this.removeThinking();
        }
    }

    updateStatus(status) {
        const statusEl = document.querySelector('.status-dot');
        if (statusEl) {
            statusEl.title = status;
        }
    }

    // ============================================================
    // MODE & BACKEND
    // ============================================================

    setMode(mode) {
        this.currentMode = mode;
        
        this.modeBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        const modeNames = {
            'chat': 'Chat Libera',
            'analisi': 'Analisi Documenti',
            'vision': 'Analisi Immagini'
        };
        
        this.updateStatus(modeNames[mode] || mode);
    }

    setBackend(backend) {
        this.currentBackend = backend;
        this.backendSelector.value = backend;
    }

    // ============================================================
    // SIDEBAR
    // ============================================================

    toggleSidebar() {
        this.sidebar.classList.toggle('open');
    }

    newChat() {
        this.saveChatHistory();
        this.showWelcome();
        this.chatTitle.textContent = 'Nuova Conversazione';
        this.setMode('chat');
        this.removeAllFiles();
        this.conversationContext = "";
        this.updateStatus('Pronto');
        this.chatId = Date.now().toString();
        this.saveChatId();
        
        if (window.innerWidth <= 768) {
            this.sidebar.classList.remove('open');
        }
    }

    // ============================================================
    // CHAT HISTORY (Local Storage)
    // ============================================================

    saveChatHistory() {
        try {
            const chatData = {
                id: this.chatId,
                messages: this.messages,
                mode: this.currentMode,
                backend: this.currentBackend,
                files: this.uploadedFiles, // Save uploaded files info
                timestamp: Date.now()
            };
            localStorage.setItem('agenteia_current_chat', JSON.stringify(chatData));
        } catch (e) {
            console.warn('Could not save chat:', e);
        }
    }

    loadChatHistory() {
        try {
            const saved = localStorage.getItem('agenteia_current_chat');
            if (saved) {
                const chatData = JSON.parse(saved);
                this.chatId = chatData.id || Date.now().toString();
                this.messages = chatData.messages || [];
                this.currentMode = chatData.mode || 'chat';
                this.currentBackend = chatData.backend || 'auto';
                this.uploadedFiles = chatData.files || [];
                
                // Restore UI
                this.backendSelector.value = this.currentBackend;
                this.setMode(this.currentMode);
                this.updateFilesDisplay();
                
                // Render messages
                if (this.messages.length > 0) {
                    this.hideWelcome();
                    this.messagesContainer.innerHTML = '';
                    this.messages.forEach(msg => {
                        const messageEl = document.createElement('div');
                        messageEl.className = `message ${msg.role}`;
                        const avatarIcon = msg.role === 'user' ? 'U' : (msg.role === 'system' ? '📄' : '🤓');
                        messageEl.innerHTML = `
                            <div class="message-avatar">${avatarIcon}</div>
                            <div class="message-content">
                                <div class="message-bubble">${this.formatContent(msg.content)}</div>
                            </div>
                        `;
                        this.messagesContainer.appendChild(messageEl);
                    });
                    this.scrollToBottom();
                }
            }
        } catch (e) {
            console.warn('Could not load chat:', e);
        }
    }

    loadChatId() {
        try {
            const saved = localStorage.getItem('agenteia_current_chat');
            if (saved) {
                const chatData = JSON.parse(saved);
                return chatData.id || Date.now().toString();
            }
        } catch (e) {}
        return Date.now().toString();
    }

    saveChatId() {
        try {
            localStorage.setItem('agenteia_chat_id', this.chatId);
        } catch (e) {}
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.agenteIA = new AgenteIA();
});
