// ============================================================
// AGENTEIA - Modern Chat Interface
// ============================================================

class AgenteIA {
    constructor() {
        this.messages = [];
        this.currentMode = 'chat';
        this.currentBackend = 'auto';
        this.isLoading = false;
        this.attachedFile = null;
        
        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.updateStatus('Pronto');
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
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // File upload
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.removeFileBtn.addEventListener('click', () => this.removeFile());

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
    }

    // ============================================================
    // CORE FUNCTIONS
    // ============================================================

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message && !this.attachedFile) return;
        if (this.isLoading) return;

        this.hideWelcome();
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.setLoading(true);

        try {
            const formData = new FormData();
            formData.append('message', message);
            formData.append('modalita', this.currentMode);
            formData.append('backend', this.currentBackend);

            if (this.attachedFile) {
                formData.append('file', this.attachedFile);
            }

            const response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.addMessage(data.response || data.message || 'Risposta ricevuta', 'assistant');

        } catch (error) {
            console.error('Error:', error);
            this.addMessage(`Errore: ${error.message}`, 'assistant');
        } finally {
            this.setLoading(false);
            this.removeFile();
        }
    }

    addMessage(content, role) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        
        const avatarIcon = role === 'user' ? 'U' : 'AI';
        
        messageEl.innerHTML = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-content">
                <div class="message-bubble">${this.formatContent(content)}</div>
            </div>
        `;

        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }

    addThinking() {
        const thinkingEl = document.createElement('div');
        thinkingEl.className = 'message assistant';
        thinkingEl.id = 'thinking-message';
        thinkingEl.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="thinking">
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span>Sto pensando...</span>
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
            'chat': 'Chat',
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
    // FILE HANDLING
    // ============================================================

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.attachedFile = file;
        
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.previewImage.src = e.target.result;
                this.previewImage.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            this.previewImage.style.display = 'none';
        }

        this.previewFilename.textContent = file.name;
        this.filePreview.style.display = 'flex';
    }

    removeFile() {
        this.attachedFile = null;
        this.fileInput.value = '';
        this.filePreview.style.display = 'none';
    }

    // ============================================================
    // SIDEBAR
    // ============================================================

    toggleSidebar() {
        this.sidebar.classList.toggle('open');
    }

    newChat() {
        this.showWelcome();
        this.chatTitle.textContent = 'Nuova Conversazione';
        this.setMode('chat');
        this.removeFile();
        this.updateStatus('Pronto');
        
        if (window.innerWidth <= 768) {
            this.sidebar.classList.remove('open');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.agenteIA = new AgenteIA();
});
