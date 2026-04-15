const { useState, useEffect, useRef } = React;

// Configuration
const API_BASE = "http://localhost:8000/api";

// Configure marked for safe rendering
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
}

// Safely render markdown to HTML
function renderMarkdown(text) {
    if (!text) return '';
    try {
        if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
            const rawHtml = marked.parse(text);
            return DOMPurify.sanitize(rawHtml);
        }
    } catch (e) {
        console.warn('Markdown parsing failed:', e);
    }
    // Fallback: basic formatting
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br/>');
}

// Utility function to show toast notifications
let toastTimeout;
const showToast = (message, duration = 3000) => {
    clearTimeout(toastTimeout);
};

// ============================================================================
// Components
// ============================================================================

function ChatMessage({ message }) {
    const isUser = message.role === "user";

    if (isUser) {
        return (
            <div className="message user-message">
                <div className="message-content">
                    {message.content}
                </div>
            </div>
        );
    }

    // Assistant: render markdown
    const html = renderMarkdown(message.content);
    return (
        <div className="message assistant-message">
            <div
                className="message-content markdown-body"
                dangerouslySetInnerHTML={{ __html: html }}
            />
        </div>
    );
}

function DocumentItem({ doc, onDelete }) {
    return (
        <div className="document-item">
            <span className="doc-name">📄 {doc}</span>
            <button 
                className="btn-delete"
                onClick={() => onDelete(doc)}
                title="Delete this document"
            >
                ✕
            </button>
        </div>
    );
}

function ChatInterface({ documents }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [useRAG, setUseRAG] = useState(true);
    const [useHybrid, setUseHybrid] = useState(true);
    const [numResults, setNumResults] = useState(5);
    const [temperature, setTemperature] = useState(0.7);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        // Add user message
        const userMessage = { role: "user", content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: input,
                    use_rag: useRAG,
                    use_hybrid: useHybrid,
                    num_results: numResults,
                    temperature: temperature,
                    chat_history: messages
                })
            });

            if (!response.ok) throw new Error("Failed to get response");

            const data = await response.json();
            const assistantMessage = { role: "assistant", content: data.response };
            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error("Error:", error);
            const errorMessage = { role: "assistant", content: `⚠️ **Error:** ${error.message}\n\nPlease check that the backend is running and try again.` };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            handleSendMessage(e);
        }
    };

    return (
        <div className="chat-interface">
            <div className="chat-header">
                <h2>💬 Chat with Your Documents</h2>
            </div>

            <div className="chat-settings">
                <label className="setting-group" id="rag-toggle">
                    <input 
                        type="checkbox" 
                        checked={useRAG} 
                        onChange={(e) => setUseRAG(e.target.checked)}
                    />
                    <span>📚 RAG Mode</span>
                </label>

                <label className="setting-group" id="hybrid-toggle">
                    <input 
                        type="checkbox" 
                        checked={useHybrid} 
                        onChange={(e) => setUseHybrid(e.target.checked)}
                        disabled={!useRAG}
                    />
                    <span>🔀 Hybrid Search</span>
                </label>

                <label className="setting-group" id="context-slider">
                    <span>📊 Context: {numResults}</span>
                    <input 
                        type="range" 
                        min="1" 
                        max="10" 
                        value={numResults}
                        onChange={(e) => setNumResults(parseInt(e.target.value))}
                    />
                </label>

                <label className="setting-group" id="temperature-slider">
                    <span>🌡️ Temp: {temperature.toFixed(1)}</span>
                    <input 
                        type="range" 
                        min="0" 
                        max="1" 
                        step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    />
                </label>
            </div>

            <div className="messages-container" id="messages-container">
                {messages.length === 0 && (
                    <div className="welcome-message">
                        <span className="welcome-icon">🤖</span>
                        <h3>Welcome to AI RAG Chatbot</h3>
                        <p>{documents.length > 0 
                            ? `I have access to ${documents.length} document(s). Ask me anything about your documents!` 
                            : "Upload a PDF document first, then ask me questions about it!"}</p>
                    </div>
                )}
                {messages.map((msg, idx) => (
                    <ChatMessage key={idx} message={msg} />
                ))}
                {loading && (
                    <div className="loading-spinner">
                        <span>🤖 Thinking</span>
                        <div className="loading-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <form className="input-form" onSubmit={handleSendMessage} id="chat-input-form">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask anything about your documents..."
                    disabled={loading}
                    id="chat-input"
                />
                <button type="submit" disabled={loading || !input.trim()} className="btn-send" id="send-button">
                    Send ➤
                </button>
            </form>
        </div>
    );
}

function DocumentUpload({ onUploadSuccess, onUploadError }) {
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const fileInputRef = useRef(null);

    const handleFileSelect = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.name.toLowerCase().endsWith(".pdf")) {
            onUploadError("❌ Only PDF files are supported");
            return;
        }

        setUploading(true);
        setProgress(10);
        const formData = new FormData();
        formData.append("file", file);

        try {
            setProgress(30);
            const response = await fetch(`${API_BASE}/documents/upload`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Upload failed");
            }
            
            setProgress(70);
            const data = await response.json();
            
            // Wait for backend to fully process
            await new Promise(resolve => setTimeout(resolve, 1500));
            setProgress(100);
            
            onUploadSuccess(`✅ ${file.name} uploaded and indexed!`);
            
            // Reset after brief delay
            setTimeout(() => {
                setProgress(0);
            }, 1000);
        } catch (error) {
            onUploadError(`❌ ${error.message}`);
            setProgress(0);
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    return (
        <div className="upload-section">
            <h3>📤 Upload Document</h3>
            <div 
                className={`upload-area ${uploading ? 'uploading' : ''}`}
                onClick={() => !uploading && fileInputRef.current.click()}
                onDragOver={(e) => !uploading && e.preventDefault()}
                onDrop={(e) => {
                    if (uploading) return;
                    e.preventDefault();
                    const files = e.dataTransfer.files;
                    if (files[0]) {
                        fileInputRef.current.files = files;
                        handleFileSelect({ target: { files } });
                    }
                }}
                style={{ opacity: uploading ? 0.7 : 1 }}
                id="upload-drop-zone"
            >
                {uploading ? (
                    <>
                        <p>⏳ Processing: {progress}%</p>
                        <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                        </div>
                    </>
                ) : (
                    <>
                        <p>📁 Drag PDF here or click to browse</p>
                        <p className="text-small">Maximum 50MB PDF files</p>
                    </>
                )}
            </div>
            <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileSelect}
                disabled={uploading}
                style={{ display: "none" }}
                id="file-input"
            />
        </div>
    );
}

function DocumentManager({ refreshTrigger, onDocumentsLoaded }) {
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState("");

    const fetchDocuments = async () => {
        try {
            setLoading(true);
            const response = await fetch(`${API_BASE}/documents`);
            if (!response.ok) throw new Error("Failed to fetch documents");
            const data = await response.json();
            const docNames = data.documents.map(d => d.name);
            
            // Get unique document names (they may have duplicate chunks)
            const uniqueNames = [...new Set(docNames)];
            setDocuments(uniqueNames);
            onDocumentsLoaded(uniqueNames);
        } catch (error) {
            console.error("Error:", error);
        } finally {
            setLoading(false);
        }
    };

    // Refresh when upload trigger changes
    useEffect(() => {
        fetchDocuments();
    }, [refreshTrigger]);
    
    // Initial load
    useEffect(() => {
        fetchDocuments();
    }, []);

    const handleDelete = async (docName) => {
        if (!confirm(`Delete "${docName}"?`)) return;

        try {
            const response = await fetch(`${API_BASE}/documents/${encodeURIComponent(docName)}`, {
                method: "DELETE"
            });

            if (!response.ok) throw new Error("Delete failed");
            
            setMessage(`✅ "${docName.split('/').pop()}" deleted`);
            await fetchDocuments();
            setTimeout(() => setMessage(""), 3000);
        } catch (error) {
            setMessage(`❌ Failed to delete: ${error.message}`);
            setTimeout(() => setMessage(""), 3000);
        }
    };

    return (
        <div className="document-manager">
            <div className="manager-header">
                <h3>📚 Documents ({documents.length})</h3>
                <button className="btn-refresh" onClick={fetchDocuments} disabled={loading} title="Refresh list" id="refresh-docs-btn">
                    🔄
                </button>
            </div>
            {message && <div className="message-notification">{message}</div>}
            
            <div className="documents-list" id="documents-list">
                {loading ? (
                    <p className="text-muted">⏳ Loading...</p>
                ) : documents.length === 0 ? (
                    <p className="text-muted">📭 No documents yet</p>
                ) : (
                    documents.map(doc => (
                        <DocumentItem 
                            key={doc} 
                            doc={doc} 
                            onDelete={handleDelete}
                        />
                    ))
                )}
            </div>
        </div>
    );
}

function Sidebar() {
    const [stats, setStats] = useState(null);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await fetch(`${API_BASE}/system/stats`);
                const data = await response.json();
                setStats(data);
            } catch (error) {
                console.error("Error fetching stats:", error);
            }
        };

        fetchStats();
        const interval = setInterval(fetchStats, 10000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <h1>🤖 AI RAG</h1>
                <p>Intelligent Document Chat</p>
            </div>

            {stats && (
                <div className="stats">
                    <div className="stat-item">
                        <span className="stat-label">📄 Documents</span>
                        <span className="stat-value">{stats.indexed_documents}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">💾 Vector DB</span>
                        <span className="stat-value">{stats.vector_db}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">🧠 LLM</span>
                        <span className="stat-value">{stats.llm_provider}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">🔍 Search</span>
                        <span className="stat-value">Hybrid</span>
                    </div>
                </div>
            )}

            <div className="sidebar-footer">
                <p>© 2026 AI RAG Application</p>
                <p className="text-small">Built with ❤️ using React, FastAPI & Groq</p>
            </div>
        </div>
    );
}

function App() {
    const [uploadRefresh, setUploadRefresh] = useState(0);
    const [documents, setDocuments] = useState([]);
    const [notification, setNotification] = useState("");

    const handleUploadSuccess = (msg) => {
        setNotification(msg);
        setUploadRefresh(prev => prev + 1);
        setTimeout(() => setNotification(""), 3000);
    };

    const handleUploadError = (msg) => {
        setNotification(msg);
        setTimeout(() => setNotification(""), 3000);
    };

    return (
        <div className="app-container">
            <Sidebar />

            <div className="main-content">
                <div className="left-panel">
                    <DocumentUpload 
                        onUploadSuccess={handleUploadSuccess}
                        onUploadError={handleUploadError}
                    />
                    {notification && <div className="notification">{notification}</div>}
                    <DocumentManager 
                        refreshTrigger={uploadRefresh}
                        onDocumentsLoaded={setDocuments}
                    />
                </div>

                <div className="right-panel">
                    <ChatInterface documents={documents} />
                </div>
            </div>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById("root"));
