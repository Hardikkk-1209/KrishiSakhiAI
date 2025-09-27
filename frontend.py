import streamlit as st
import requests
import json
from typing import Dict, List
import numpy as np
import faiss
import pickle
from pathlib import Path
import base64
import time

# Configure Streamlit page
st.set_page_config(
    page_title="KrishiSakhiAI - Smart Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- START: Premium Modern CSS with Animations ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary-green: #059669;
    --primary-green-dark: #047857;
    --secondary-green: #10b981;
    --accent-blue: #3b82f6;
    --dark-bg: #0f172a;
    --darker-bg: #020617;
    --card-bg: rgba(30, 41, 59, 0.8);
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --border-subtle: rgba(148, 163, 184, 0.1);
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    transition: all 0.3s ease;
}

.stApp {
    background: linear-gradient(135deg, var(--darker-bg) 0%, var(--dark-bg) 50%, #1e293b 100%);
    color: var(--text-primary);
}

/* Animated Background Particles */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(34, 197, 94, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
}

/* Main Header with Glass Morphism */
.main-header {
    background: linear-gradient(135deg, 
        rgba(5, 150, 105, 0.9) 0%, 
        rgba(16, 185, 129, 0.8) 50%, 
        rgba(34, 197, 94, 0.7) 100%
    );
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 3rem 2rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: 
        0 20px 40px rgba(16, 185, 129, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.main-header h1 {
    text-align: center;
    font-size: 4rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    background: linear-gradient(135deg, #fff, #e0f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-header p {
    text-align: center;
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.3rem;
    font-weight: 400;
    margin-top: 1rem;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Enhanced Chat Messages */
.stChatMessage {
    background: var(--card-bg) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    margin: 1.5rem 0 !important;
    border: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    position: relative;
    overflow: hidden;
}

.stChatMessage::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--secondary-green), var(--accent-blue));
    opacity: 0;
    transition: opacity 0.3s ease;
}

.stChatMessage:hover::before {
    opacity: 1;
}

.stChatMessage[data-testid*="user"] {
    border-left: 4px solid var(--secondary-green);
    background: linear-gradient(135deg, 
        rgba(16, 185, 129, 0.1) 0%, 
        var(--card-bg) 100%
    ) !important;
}

.stChatMessage[data-testid*="assistant"] {
    border-left: 4px solid var(--accent-blue);
    background: linear-gradient(135deg, 
        rgba(59, 130, 246, 0.1) 0%, 
        var(--card-bg) 100%
    ) !important;
}

/* Modern Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-green), var(--secondary-green)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    transform: translateY(0);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--primary-green-dark), var(--primary-green)) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.4);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
}

/* Enhanced Chat Input - Green Theme */
.stChatInput > div > div {
    background: var(--card-bg) !important;
    border: 2px solid var(--secondary-green) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.stChatInput > div > div:focus-within {
    border-color: var(--primary-green) !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
}

/* Sidebar Enhancements */
.css-1d391kg {
    background: linear-gradient(180deg, var(--darker-bg) 0%, var(--dark-bg) 100%);
    border-right: 1px solid var(--border-subtle);
}

.sidebar-content {
    padding: 1rem;
}

/* Status Indicators */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

.status-success {
    background: rgba(34, 197, 94, 0.1);
    color: var(--success);
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-error {
    background: rgba(239, 68, 68, 0.1);
    color: var(--error);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-info {
    background: rgba(59, 130, 246, 0.1);
    color: var(--accent-blue);
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.status-warning {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Enhanced Selectbox and Sliders */
.stSelectbox > div > div {
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    color: var(--text-primary);
}

.stSlider > div > div > div > div {
    background: var(--secondary-green);
}

/* File Uploader */
.stFileUploader > div {
    background: var(--card-bg);
    border: 2px dashed var(--border-subtle);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: var(--secondary-green);
    background: rgba(16, 185, 129, 0.05);
}

/* Expandable Sections */
.streamlit-expanderHeader {
    background: var(--card-bg);
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
}

/* Metrics and Stats - Improved Single Line Design */
.metric-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), var(--card-bg));
    padding: 1rem 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: all 0.3s ease;
    min-height: 60px;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    border-color: var(--secondary-green);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--secondary-green);
    margin: 0;
    line-height: 1;
}

.metric-label {
    font-size: 0.95rem;
    color: var(--text-primary);
    font-weight: 500;
    margin: 0;
    line-height: 1;
}

/* Loading Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading-text {
    animation: pulse 2s infinite;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--darker-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--secondary-green);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-green);
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
    }
    
    .stChatMessage {
        margin: 1rem 0 !important;
        padding: 1rem !important;
    }
}

/* Hover Effects for Interactive Elements */
.stCheckbox:hover,
.stSelectbox:hover,
.stSlider:hover {
    transform: translateY(-1px);
}

/* Focus States */
.stChatInput input:focus {
    outline: none;
    border-color: var(--secondary-green) !important;
}

/* Welcome Box for Front Page */
.welcome-box {
    background: linear-gradient(135deg, 
        rgba(16, 185, 129, 0.15) 0%, 
        rgba(5, 150, 105, 0.1) 50%, 
        rgba(34, 197, 94, 0.05) 100%
    );
    border: 2px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    padding: 2rem 1.5rem;
    margin: 1rem 0;
    text-align: center;
    backdrop-filter: blur(15px);
    box-shadow: 0 15px 30px rgba(16, 185, 129, 0.1);
    position: relative;
    overflow: hidden;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

.welcome-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(16, 185, 129, 0.05), transparent);
    animation: shimmer 4s infinite;
}

.welcome-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--secondary-green);
    margin-bottom: 1rem;
    text-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
}

.welcome-message {
    font-size: 1.1rem;
    color: var(--text-primary);
    line-height: 1.5;
    margin-bottom: 1.5rem;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}

.topic-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    justify-content: center;
    margin-top: 1.5rem;
}

.topic-button {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    color: var(--text-primary);
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

.topic-button:hover {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(5, 150, 105, 0.2));
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);
}

/* Enhanced Spinner */
.stSpinner > div {
    border-color: var(--secondary-green) !important;
}

</style>
""", unsafe_allow_html=True)
# --- END: Premium Modern CSS ---

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Ollama + FAISS Configuration
OLLAMA_URL = "http://localhost:11434"
VECTOR_STORE_DIR = "agricultural_vector_store"

class VectorStoreManager:
    def __init__(self, vector_store_dir: str):
        self.vector_store_dir = Path(vector_store_dir)
        self.index = None
        self.documents = []
        self.chunks_metadata = []
        self.dimension = 768
        self.loaded = False

    def load_vector_store(self):
        try:
            index_path = self.vector_store_dir / "faiss_index.bin"
            metadata_path = self.vector_store_dir / "metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                return False
                
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.chunks_metadata = data['chunks_metadata']
                self.dimension = data['dimension']
            
            self.loaded = True
            return True
        except Exception as e:
            st.error(f"❌ Error loading vector store: {e}")
            return False

    def get_embedding(self, text: str, model_name="nomic-embed-text"):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": model_name, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            st.error(f"🔗 Embedding error: {e}")
            return None

    def search_similar_documents(self, query: str, k=3):
        if not self.loaded:
            return []
        
        embedding = self.get_embedding(query)
        if embedding is None:
            return []
        
        query_array = np.array([embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        scores, indices = self.index.search(query_array, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score > 0.1:
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.chunks_metadata[idx],
                    "score": float(score)
                })
        return results

# Enhanced System Prompt
SYSTEM_PROMPT = """You are KrishiSakhi, an advanced AI agricultural assistant with deep expertise in farming practices.

You provide:
✅ Practical, actionable advice for crop management
✅ Pest and disease identification and treatment
✅ Weather-based farming recommendations  
✅ Sustainable and organic farming methods
✅ Cost-effective solutions for small and large farmers
✅ Soil health and irrigation guidance
✅ Market trends and crop economics

Always prioritize:
- Farmer safety and environmental sustainability
- Cost-effectiveness and resource efficiency
- Local climate and soil conditions
- Simple, implementable solutions

Communicate in a warm, supportive manner using clear, practical language that farmers can easily understand and apply.
"""

def get_available_models():
    """Fetch available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except:
        return []

def create_enhanced_prompt(user_query: str, retrieved_docs: List[Dict]):
    """Create context-enhanced prompt with retrieved documents"""
    context = ""
    if retrieved_docs:
        context = "\n📚 **Relevant Knowledge Base Information:**\n"
        for i, doc in enumerate(retrieved_docs, 1):
            filename = doc['metadata'].get('filename', 'Unknown')
            content_preview = doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content']
            context += f"\n**[Source {i} - {filename}]:**\n{content_preview}\n"
    
    enhanced_prompt = f"{context}\n\n🌾 **Farmer's Question:** {user_query}"
    return enhanced_prompt

def stream_chat_response(model: str, messages: List[Dict], temperature: float = 0.7):
    """Stream chat response from Ollama"""
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "system": SYSTEM_PROMPT
        }
    }
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode())
                            if "message" in chunk and "content" in chunk["message"]:
                                yield chunk["message"]["content"]
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"❌ Server Error: {response.status_code}"
    except Exception as e:
        yield f"❌ Connection Error: {str(e)}"

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def render_status_indicator(status_type: str, message: str, icon: str = ""):
    """Render a modern status indicator"""
    return f"""
    <div class="status-indicator status-{status_type}">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager(VECTOR_STORE_DIR)
if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

# Header removed as requested

# Enhanced Sidebar with Modern Design
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    
    # Check Ollama connection
    ollama_connected = check_ollama_connection()
    
    if ollama_connected:
        st.markdown(render_status_indicator("success", "Ollama Connected", "🟢"), unsafe_allow_html=True)
        
        # Load vector store if not already loaded
        if not st.session_state.vector_store.loaded:
            with st.spinner("🔄 Loading agricultural knowledge base..."):
                if st.session_state.vector_store.load_vector_store():
                    st.markdown(render_status_indicator("info", "Knowledge Base Ready", "📚"), unsafe_allow_html=True)
                else:
                    st.markdown(render_status_indicator("warning", "Knowledge Base Not Found", "⚠️"), unsafe_allow_html=True)
        else:
            st.markdown(render_status_indicator("info", "Knowledge Base Ready", "📚"), unsafe_allow_html=True)
        
        # Model selection - Filter out embedding models
        available_models = get_available_models()
        # Filter out embedding models from the UI
        chat_models = [model for model in available_models if not any(embed_keyword in model.lower() for embed_keyword in ['embed', 'embedding', 'nomic-embed'])]
        
        if chat_models:
            default_model = None
            if "llama3.2" in chat_models:
                default_model = chat_models.index("llama3.2")
            elif chat_models:
                default_model = 0
                
            st.session_state.selected_model = st.selectbox(
                "🤖 Choose AI Model",
                chat_models,
                index=default_model,
                help="Select the language model for conversations"
            )
            
            # Advanced settings
            st.markdown("### 🔧 Advanced Settings")
            
            use_knowledge_base = st.checkbox(
                "📖 Enable Knowledge Base Search", 
                value=True,
                help="Use agricultural knowledge base for enhanced responses"
            )
            
            if use_knowledge_base:
                k_documents = st.slider(
                    "📄 Context Documents", 
                    min_value=1, 
                    max_value=5, 
                    value=3,
                    help="Number of relevant documents to include"
                )
            else:
                k_documents = 0
                
            temperature = st.slider(
                "🎨 Response Creativity", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                help="Lower values = more focused, Higher values = more creative"
            )
            
            # Stats section
            st.markdown("### 📊 Session Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.messages)}</div>
                    <div class="metric-label">Messages</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.conversation_count}</div>
                    <div class="metric-label">Queries</div>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.markdown(render_status_indicator("error", "No Chat Models Available", "❌"), unsafe_allow_html=True)
            st.markdown("""
            **Setup Instructions:**
            ```bash
            # Install required models
            ollama pull llama3.2
            ollama pull llava
            ollama pull nomic-embed-text
            ```
            """)
    else:
        st.markdown(render_status_indicator("error", "Ollama Offline", "🔴"), unsafe_allow_html=True)
        st.markdown("""
        **To start Ollama:**
        ```bash
        ollama serve
        ```
        """)
    
    # Clear chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.rerun()

# Main Chat Interface
chat_container = st.container()

# Show welcome message only if no conversation has started
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-box">
        <h2 class="welcome-title">🙏 Namaste! Welcome to KrishiSakhiAI!</h2>
        <p class="welcome-message">
            Hello, friend! I'm KrishiSakhi, your helpful agricultural assistant. I'm here 
            to provide you with expert advice and support for all your farming needs.
        </p>
        <p class="welcome-message">
            Feel free to ask me anything about:
        </p>
        <div class="topic-buttons">
            <div class="topic-button">🌱 Crop Management</div>
            <div class="topic-button">🐛 Pest and Disease Control</div>
            <div class="topic-button">💧 Irrigation</div>
            <div class="topic-button">💰 Farm Economics</div>
        </div>
        <p class="welcome-message" style="margin-top: 2rem; font-size: 1.1rem;">
            Let's start! Just type your question below.
        </p>
    </div>
    """, unsafe_allow_html=True)

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display uploaded image if present
            if message["role"] == "user" and "image" in message:
                st.image(message["image"], width=200, caption="Uploaded Image")
            
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Knowledge Base Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        filename = source["metadata"].get("filename", "Unknown")
                        score = source.get("score", 0)
                        st.markdown(f"**{i}. {filename}** (Relevance: {score:.3f})")
                        st.markdown(f"```\n{source['content'][:200]}...\n```")

# Image Upload Section
uploaded_image = st.file_uploader(
    "📷 Upload Crop/Plant Image (Optional)",
    type=["png", "jpg", "jpeg", "webp"],
    help="Upload images of crops, plants, pests, or diseases for visual analysis"
)

# Enhanced Chat Input
if prompt := st.chat_input("🌱 Ask KrishiSakhi about farming, crops, pests, weather, or any agricultural topic..."):
    # Handle image upload
    image_bytes = None
    base64_image = None
    
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        base64_image = encode_image(image_bytes)

    # Create user message
    user_message = {"role": "user", "content": prompt}
    if image_bytes:
        user_message["image"] = image_bytes
        user_message["base64_image"] = base64_image

    st.session_state.messages.append(user_message)
    st.session_state.conversation_count += 1

    # Display user message
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=200, caption="Uploaded Image")
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Auto-switch to vision model if image is uploaded
        selected_model = st.session_state.selected_model
        if image_bytes and "llava" not in selected_model.lower():
            # Try to find a vision-capable model
            vision_models = [m for m in get_available_models() if "llava" in m.lower()]
            if vision_models:
                selected_model = vision_models[0]
                st.info(f"🔍 Image detected → Switching to **{selected_model}** for visual analysis")
            else:
                st.warning("⚠️ No vision model available. Install llava: `ollama pull llava`")

        # Retrieve relevant documents
        retrieved_documents = []
        if use_knowledge_base and st.session_state.vector_store.loaded:
            with st.spinner("🔍 Searching knowledge base..."):
                retrieved_documents = st.session_state.vector_store.search_similar_documents(
                    prompt, k_documents
                )

        # Create enhanced prompt with context
        enhanced_prompt = create_enhanced_prompt(prompt, retrieved_documents)
        
        # Prepare messages for Ollama API
        ollama_messages = []
        for msg in st.session_state.messages:
            content = enhanced_prompt if msg == user_message else msg["content"]
            ollama_msg = {"role": msg["role"], "content": content}
            
            # Add image for vision models
            if "base64_image" in msg and msg["role"] == "user":
                ollama_msg["images"] = [msg["base64_image"]]
                
            ollama_messages.append(ollama_msg)

        # Stream the response
        for chunk in stream_chat_response(selected_model, ollama_messages, temperature):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        # Final response without cursor
        message_placeholder.markdown(full_response)

        # Save assistant response
        assistant_message = {"role": "assistant", "content": full_response}
        if retrieved_documents:
            assistant_message["sources"] = retrieved_documents
            
        st.session_state.messages.append(assistant_message)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748b;">
    <p><strong>KrishiSakhiAI</strong> - Empowering farmers with AI-powered agricultural insights</p>
    <p style="font-size: 0.9rem;">Built with ❤️ for the farming community | Powered by Ollama & Streamlit</p>
</div>
""", unsafe_allow_html=True)