import streamlit as st
import requests
import json
from typing import Dict, List, Generator
import time
import numpy as np
import faiss
import pickle
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="KrishiSakhiAI Chat",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- START: Enhanced Custom CSS ---
st.markdown("""
<style>
    /* Import Google Fonts for a modern, clean look */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main theme colors - Dark agricultural theme */
    .stApp {
        background: linear-gradient(135deg, #0e161c 0%, #172a3a 50%, #203e57 100%);
        color: #e3f2fd;
    }
    
    /* Header styling with a dynamic background */
    .main-header {
        background: linear-gradient(135deg, #065f46, #059669, #10b981);
        padding: 3rem 1rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.4);
        border: 1px solid rgba(16, 185, 129, 0.3);
        animation: gradient-anim 5s ease infinite;
    }

    @keyframes gradient-anim {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .main-header h1 {
        color: #ffffff;
        text-align: center;
        margin: 0;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        letter-spacing: 2px;
    }
    
    .main-header p {
        color: #d1fae5;
        text-align: center;
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #064e3b 0%, #065f46 100%);
    }
    
    .sidebar-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    /* Streamlit sidebar text color fix */
    .css-1d391kg .stMarkdown, .css-1d391kg label {
        color: #ffffff !important;
    }
    
    /* Status indicators */
    .status-box {
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        animation: status-pulse 1.5s infinite;
    }
    
    @keyframes status-pulse {
        0% {box-shadow: 0 4px 12px rgba(0,0,0,0.2);}
        50% {box-shadow: 0 4px 18px rgba(0,0,0,0.4);}
        100% {box-shadow: 0 4px 12px rgba(0,0,0,0.2);}
    }
    
    .status-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }
    
    /* Chat message styling - Much improved */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* User message styling */
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, rgba(34, 139, 34, 0.15), rgba(46, 139, 87, 0.15)) !important;
        border-left: 5px solid #10b981;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid*="assistant"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6)) !important;
        border-left: 5px solid #3b82f6;
    }
    
    /* Message text color and typography */
    .stChatMessage .stMarkdown {
        color: #e3f2fd !important;
    }
    
    .stChatMessage .stMarkdown p {
        color: #e3f2fd !important;
        line-height: 1.8;
        margin-bottom: 0.8rem;
    }
    
    .stChatMessage .stMarkdown h1, .stChatMessage .stMarkdown h2, .stChatMessage .stMarkdown h3 {
        color: #10b981 !important;
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
        font-weight: 600;
    }
    
    .stChatMessage .stMarkdown strong {
        color: #81c784 !important;
        font-weight: 600;
    }
    
    .stChatMessage .stMarkdown code {
        background: rgba(100, 116, 139, 0.2) !important;
        color: #fbbf24 !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'Fira Code', 'Monaco', monospace;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #047857, #059669) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Instructions card */
    .instructions-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        color: #e3f2fd;
    }
    
    /* Source citations */
    .source-citation {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
        color: #e3f2fd;
        backdrop-filter: blur(5px);
    }
    
    /* Welcome content styling */
    .welcome-content {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        color: #e3f2fd;
        text-align: center;
    }
    
    .welcome-content h3 {
        color: #10b981 !important;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .welcome-content p {
        color: #e3f2fd !important;
        line-height: 1.7;
        margin-bottom: 1.2rem;
        font-size: 1.1rem;
    }
    
    .welcome-content ul {
        list-style-type: none;
        padding: 0;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem;
    }
    
    .welcome-content li {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 0.7rem 1.2rem;
        color: #e3f2fd;
        font-weight: 500;
        transition: transform 0.2s ease;
    }
    
    .welcome-content li:hover {
        transform: scale(1.05);
    }
    
    /* Chat input styling */
    .stChatInput > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput input {
        background: transparent !important;
        color: #e3f2fd !important;
        border: none !important;
        font-size: 1.1rem;
    }
    
    .stChatInput input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #064e3b, #065f46);
        color: #d1fae5;
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        margin-top: 3rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 12px !important;
        color: #e3f2fd !important;
    }
</style>
""", unsafe_allow_html=True)
# --- END: Enhanced Custom CSS ---

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434"
VECTOR_STORE_DIR = "agricultural_vector_store"  # Directory containing your vector store

class VectorStoreManager:
    """Manages the FAISS vector store for document retrieval"""
    
    def __init__(self, vector_store_dir: str):
        self.vector_store_dir = Path(vector_store_dir)
        self.index = None
        self.documents = []
        self.chunks_metadata = []
        self.dimension = 768
        self.loaded = False
        
    def load_vector_store(self):
        """Load the FAISS vector store"""
        try:
            index_path = self.vector_store_dir / "faiss_index.bin"
            metadata_path = self.vector_store_dir / "metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                st.error(f"Vector store not found in {self.vector_store_dir}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.chunks_metadata = data['chunks_metadata']
                self.dimension = data['dimension']
            
            self.loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return False
    
    def get_embedding(self, text: str, model_name: str = "nomic-embed-text"):
        """Get embedding for text using Ollama"""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": model_name,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return None
    
    def search_similar_documents(self, query: str, k: int = 3):
        """Search for similar documents in the vector store"""
        if not self.loaded:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        try:
            # Normalize query embedding
            query_array = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score > 0.1:  # Threshold for relevance
                    results.append({
                        'content': self.documents[idx],
                        'metadata': self.chunks_metadata[idx],
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error searching vector store: {e}")
            return []

# System prompt for the agricultural assistant
SYSTEM_PROMPT = """You are KrishiSakhi, a knowledgeable and friendly agricultural assistant and farmer's best friend. You have deep expertise in farming, agriculture, and rural development. Your personality is warm, supportive, and practical.

Your expertise includes:
🌾 **Crop Management**: Planting schedules, crop selection, rotation strategies, harvesting techniques
🐛 **Pest & Disease Control**: Identification, prevention, organic and chemical treatment options
🌱 **Seasonal Farming**: Crop recommendations for different seasons (Kharif, Rabi, Zaid)
🌦️ **Weather & Climate**: Impact analysis, adaptation strategies, climate-resilient farming
💧 **Irrigation & Water Management**: Efficient watering, drip irrigation, water conservation
🌿 **Sustainable Practices**: Organic farming, soil health, composting, natural fertilizers
💰 **Farm Economics**: Cost optimization, market trends, profit maximization, subsidies
🚜 **Farm Equipment**: Machinery recommendations, maintenance, modern farming tools
🌾 **Post-Harvest**: Storage techniques, value addition, processing, marketing

**Your communication style:**
- Address farmers as "friend" or "bhai/behan" occasionally
- Use simple, practical language that farmers can easily understand
- Provide actionable advice with specific steps
- Consider local farming conditions and traditional knowledge
- Be encouraging and supportive, especially during challenges
- Include relevant seasonal timing and regional considerations
- Mention specific crop varieties, fertilizer quantities, and treatment dosages when appropriate

**Important guidelines:**
- Always prioritize farmer safety when recommending pesticides or chemicals
- Suggest integrated pest management approaches
- Consider cost-effectiveness for small farmers
- Recommend both traditional and modern approaches when relevant
- Encourage sustainable and environmentally friendly practices

When using information from the knowledge base, integrate it naturally into your responses and cite the sources when providing specific data or recommendations.

Remember, you're not just an AI - you're a trusted friend who understands the challenges of farming life and wants to help farmers succeed and thrive."""

def get_available_models() -> List[str]:
    """Fetch available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()
            return [model["name"] for model in models.get("models", [])]
        else:
            return []
    except requests.exceptions.RequestException:
        return []

def create_enhanced_prompt(user_query: str, context_docs: List[Dict]) -> str:
    """Create enhanced prompt with context from vector store"""
    context_text = ""
    
    if context_docs:
        context_text = "\n**Relevant Information from Knowledge Base:**\n"
        for i, doc in enumerate(context_docs, 1):
            context_text += f"\n[Source {i} - {doc['metadata']['filename']}]:\n"
            context_text += f"{doc['content'][:500]}...\n"
        
        context_text += "\n**Based on the above information and your agricultural expertise, please answer the farmer's question:**\n"
    
    enhanced_query = f"{context_text}\n**Farmer's Question:** {user_query}"
    return enhanced_query

def stream_chat_response(model: str, messages: List[Dict], temperature: float = 0.7) -> Generator[str, None, None]:
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
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'message' in chunk and 'content' in chunk['message']:
                                yield chunk['message']['content']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        yield f"Connection error: {str(e)}"

def check_ollama_connection() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager(VECTOR_STORE_DIR)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌾 KrishiSakhiAI</h1>
    <p>Your AI-Powered Agricultural Assistant with Knowledge Base</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Check Ollama connection
    if check_ollama_connection():
        st.markdown('<div class="status-box status-success">🟢 Ollama Server Connected</div>', unsafe_allow_html=True)
        
        # Check vector store
        if not st.session_state.vector_store.loaded:
            with st.spinner("Loading knowledge base..."):
                if st.session_state.vector_store.load_vector_store():
                    st.markdown('<div class="status-box status-success">📚 Knowledge Base Loaded</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="kb-info">📊 <strong>KB Info:</strong><br>📄 Chunks: {len(st.session_state.vector_store.chunks_metadata)}<br>🧠 Dim: {st.session_state.vector_store.dimension}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-box status-warning">⚠️ Knowledge Base Not Found</div>', unsafe_allow_html=True)
                    st.markdown("Please run the document indexer first.")
        else:
            st.markdown('<div class="status-box status-success">📚 Knowledge Base Ready</div>', unsafe_allow_html=True)
        
        # Get available models
        available_models = get_available_models()
        
        if available_models:
            st.markdown("### 🤖 Model Selection")
            selected_model = st.selectbox(
                "Choose AI Model:",
                available_models,
                index=0 if not st.session_state.selected_model else 
                      (available_models.index(st.session_state.selected_model) 
                       if st.session_state.selected_model in available_models else 0)
            )
            st.session_state.selected_model = selected_model
            
            # Vector search settings
            st.markdown("### 🔍 Search Settings")
            use_vector_search = st.checkbox("Enable Knowledge Base Search", value=True)
            if use_vector_search:
                search_results_count = st.slider("Context Documents", 1, 5, 3)
            else:
                search_results_count = 0
            
            # Temperature control
            st.markdown("### 🌡️ Response Settings")
            temperature = st.slider(
                "Creativity Level:",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Lower values = more focused responses, Higher values = more creative responses"
            )
            
            # Model info
            st.markdown("### 📊 Current Setup")
            st.info(f"**Active Model:** {selected_model}\n\n**Knowledge Base:** {'Enabled' if use_vector_search else 'Disabled'}\n\n**Creativity:** {temperature}")
            
        else:
            st.markdown('<div class="status-box status-warning">⚠️ No Models Available</div>', unsafe_allow_html=True)
            st.markdown("### 📥 Setup Instructions")
            st.code("ollama pull llama2", language="bash")
            st.code("ollama pull mistral", language="bash")
            
    else:
        st.markdown('<div class="status-box status-error">🔴 Ollama Server Offline</div>', unsafe_allow_html=True)
        st.markdown("### 🚀 Quick Start")
        st.code("ollama serve", language="bash")
    
    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    <div class="instructions-card">
        <h4>🌱 Getting Started</h4>
        <ol>
            <li><strong>Start Ollama:</strong> Run <code>ollama serve</code></li>
            <li><strong>Install Models:</strong> Pull your preferred AI model</li>
            <li><strong>Create Knowledge Base:</strong> Run the document indexer</li>
            <li><strong>Ask Questions:</strong> Get expert agricultural advice!</li>
        </ol>
        
        <h4>🎯 Sample Questions:</h4>
        <ul>
            <li>"What crops should I plant in Kharif season?"</li>
            <li>"How to control aphids on my tomato crop?"</li>
            <li>"Best fertilizer schedule for wheat?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Check if we can proceed
if not check_ollama_connection():
    st.error("🚫 Please start Ollama server first: `ollama serve`")
    st.markdown("""
    ### 🔧 Troubleshooting
    1. **Install Ollama:** Visit [ollama.ai](https://ollama.ai) to download
    2. **Start Server:** Run `ollama serve` in your terminal
    3. **Install Models:** Use `ollama pull llama2` or similar
    4. **Install Embedding Model:** Run `ollama pull nomic-embed-text`
    5. **Refresh Page:** Come back once Ollama is running
    """)
    st.stop()

if not st.session_state.selected_model:
    st.warning("🎯 Please select an AI model from the sidebar to start chatting")
    st.stop()

# Display chat messages
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Use columns for a more balanced welcome screen
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.markdown("""
            <div class="welcome-content">
                <h3>🙏 Namaste! Welcome to KrishiSakhiAI!</h3>
                
                
                
               
                
              
            
            """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Knowledge Base Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-citation">
                        <strong>Source {i}:</strong> {source['metadata']['filename']} 
                        (Chunk {source['metadata']['chunk_index']+1}/{source['metadata']['total_chunks']})
                        <br><small>Relevance Score: {source['score']:.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("🌾 Ask KrishiSakhi anything about farming..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Search vector store if enabled
        relevant_docs = []
        if use_vector_search and st.session_state.vector_store.loaded:
            with st.spinner("🔍 Searching knowledge base..."):
                relevant_docs = st.session_state.vector_store.search_similar_documents(
                    prompt, k=search_results_count
                )
        
        # Create enhanced prompt with context
        enhanced_prompt = create_enhanced_prompt(prompt, relevant_docs)
        
        # Convert chat history to Ollama format
        ollama_messages = []
        for msg in st.session_state.messages[:-1]:  # Exclude the current user message
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add the enhanced current message
        ollama_messages.append({
            "role": "user",
            "content": enhanced_prompt
        })
        
        # Stream the response
        try:
            for chunk in stream_chat_response(
                st.session_state.selected_model, 
                ollama_messages, 
                temperature
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history with sources
            assistant_message = {
                "role": "assistant", 
                "content": full_response
            }
            
            if relevant_docs:
                assistant_message["sources"] = relevant_docs
            
            st.session_state.messages.append(assistant_message)
            
            # Show sources
            if relevant_docs:
                with st.expander("📚 Knowledge Base Sources Used"):
                    for i, source in enumerate(relevant_docs, 1):
                        st.markdown(f"""
                        <div class="source-citation">
                        <strong>Source {i}:</strong> {source['metadata']['filename']} 
                        (Chunk {source['metadata']['chunk_index']+1}/{source['metadata']['total_chunks']})
                        <br><small>Relevance Score: {source['score']:.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; align-items: center; gap: 2rem;">
        <div>🌾 <strong>KrishiSakhiAI</strong></div>
        <div>•</div>
        <div>Powered by <strong>Ollama</strong> & <strong>FAISS Vector Store</strong></div>
        <div>•</div>
        <div>🚜 Empowering Farmers with AI & Knowledge</div>
    </div>
</div>
""", unsafe_allow_html=True)