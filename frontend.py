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
import pandas as pd
import sys
import os

# Add project root for livestock module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Livestock Biosecurity imports
try:
    from livestock_biosecurity.models import load_models as load_livestock_models, HEALTH_LABELS
    from livestock_biosecurity.cv_module import GaitAnalyzer, BehaviorAnalyzer
    from livestock_biosecurity.alert_system import BiosecurityAlertSystem
    from livestock_biosecurity.data_generator import LivestockDataGenerator
    LIVESTOCK_AVAILABLE = True
except ImportError:
    LIVESTOCK_AVAILABLE = False

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

/* ── Livestock Scanner Styles ── */
.livestock-hero {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(15px);
    position: relative;
    overflow: hidden;
}
.livestock-hero::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; right: -100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--secondary-green), transparent);
    animation: shimmer 4s ease-in-out infinite;
}
.livestock-hero h2 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff, #a7f3d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
}
.livestock-hero p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0;
}
.result-card {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 1.3rem;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
    border-color: rgba(16, 185, 129, 0.3);
}
.result-card-green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.result-card-orange::before { background: linear-gradient(90deg, #f97316, #fb923c); }
.result-card-red::before { background: linear-gradient(90deg, #ef4444, #f87171); }
.result-card-blue::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.result-title {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.result-value {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
}
.input-section-label {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--secondary-green);
    margin: 1rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.alert-banner {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid;
    font-size: 0.88rem;
}
.alert-banner-critical {
    background: rgba(239, 68, 68, 0.1);
    border-left-color: #ef4444;
    color: #fca5a5;
}
.alert-banner-warning {
    background: rgba(249, 115, 22, 0.1);
    border-left-color: #f97316;
    color: #fdba74;
}
.alert-banner-healthy {
    background: rgba(16, 185, 129, 0.1);
    border-left-color: #10b981;
    color: #6ee7b7;
}
.gait-meter-bar {
    display: flex;
    gap: 3px;
    margin: 0.5rem 0;
}
.gait-segment {
    flex: 1;
    height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
}

/* ═══ Onboarding & Registration Styles ═══ */
.onboard-container {
    max-width: 700px;
    margin: 2rem auto;
    padding: 2.5rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    border-radius: 24px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 40px rgba(16, 185, 129, 0.08);
}
.onboard-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #34d399, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.onboard-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.onboard-step {
    display: flex;
    justify-content: center;
    gap: 0.8rem;
    margin-bottom: 2rem;
}
.step-dot {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
    transition: all 0.3s ease;
}
.step-dot.active {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
}
.step-dot.done {
    background: #047857;
    color: white;
}
.step-dot.inactive {
    background: rgba(51, 65, 85, 0.6);
    color: #64748b;
    border: 1px solid rgba(100,116,139,0.3);
}
.region-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.region-card:hover {
    border-color: rgba(16, 185, 129, 0.5);
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.1);
}
.region-card h4 {
    color: #10b981;
    font-size: 1.3rem;
    margin-bottom: 0.5rem;
}
.disease-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 14px;
    padding: 1.2rem;
    margin: 0.8rem 0;
}
.disease-card h5 {
    color: #f87171;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}
.disease-section {
    margin: 0.5rem 0;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    font-size: 0.9rem;
    line-height: 1.6;
}
.disease-section.symptoms { background: rgba(251, 191, 36, 0.08); border-left: 3px solid #fbbf24; }
.disease-section.prevention { background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; }
.disease-section.treatment { background: rgba(59, 130, 246, 0.08); border-left: 3px solid #3b82f6; }
.disease-section.pesticide { background: rgba(168, 85, 247, 0.08); border-left: 3px solid #a855f7; }
.crop-hero {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08));
    border-radius: 16px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    margin-bottom: 1.5rem;
}
.crop-hero h3 {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.crop-hero p { color: #94a3b8; font-size: 0.95rem; }

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

# Enhanced System Prompt — Personalized per farmer
def get_system_prompt():
    """Build a system prompt that includes the farmer's profile for personalized answers."""
    base = """You are KrishiSakhi, an advanced AI agricultural assistant with deep expertise in farming practices.

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
    # Inject farmer profile if registered
    fp = st.session_state.get('farmer_profile', {})
    if fp:
        region = fp.get('region', 'Unknown')
        try:
            region_info = MAHARASHTRA_CROPS.get(region, {})
        except NameError:
            region_info = {}
        rare_crops = ', '.join(region_info.get('rare', [])) if region_info else 'N/A'
        common_crops = ', '.join(region_info.get('common', [])) if region_info else 'N/A'
        climate = region_info.get('climate', 'N/A')
        soil = region_info.get('soil', 'N/A')

        profile_context = f"""

--- FARMER PROFILE (use this to personalize every answer) ---
👨‍🌾 Name: {fp.get('name', 'Farmer')}
📍 Region: {region}, Maharashtra
🏞️ Land Size: {fp.get('land_size', 'N/A')} acres
💧 Water Source: {fp.get('water_source', 'N/A')}
🌱 Current Crops: {fp.get('current_crop', 'N/A')}
🌤️ Climate: {climate}
🪨 Soil Type: {soil}
🌿 Common Crops in Region: {common_crops}
🌟 Rare/Exotic Crop Opportunities: {rare_crops}

IMPORTANT: Always tailor your answers to this farmer's specific region ({region}), soil ({soil}), climate ({climate}), water source ({fp.get('water_source', '')}), and land size ({fp.get('land_size', '')} acres). When suggesting crops, pesticides, fertilizers, or practices, consider local conditions. Address the farmer by name ({fp.get('name', 'farmer')}) when appropriate. If the farmer asks about crops, prioritize those suitable for {region} region. Always mention the rare crop opportunity ({rare_crops}) when relevant.
"""
        base += profile_context
    return base

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
            "system": get_system_prompt()
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
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Chat"
if "livestock_models" not in st.session_state:
    st.session_state.livestock_models = None
if "livestock_result" not in st.session_state:
    st.session_state.livestock_result = None
# Onboarding state
if "farmer_registered" not in st.session_state:
    st.session_state.farmer_registered = False
if "onboarding_step" not in st.session_state:
    st.session_state.onboarding_step = 1
if "farmer_profile" not in st.session_state:
    st.session_state.farmer_profile = {}
if "selected_region" not in st.session_state:
    st.session_state.selected_region = None

# Load livestock models
@st.cache_resource
def _load_livestock_models():
    try:
        return load_livestock_models('trained_models')
    except Exception:
        return None

if LIVESTOCK_AVAILABLE:
    st.session_state.livestock_models = _load_livestock_models()

# Header removed as requested

# Enhanced Sidebar with Modern Design
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")

    # ── Mode Toggle ──
    if st.session_state.farmer_registered and LIVESTOCK_AVAILABLE and st.session_state.livestock_models:
        st.session_state.app_mode = st.radio(
            "📌 Mode",
            ["💬 Chat Assistant", "🐄 Livestock Health Scanner"],
            index=0 if st.session_state.app_mode == "Chat" else 1,
            help="Switch between the AI Chat and Livestock Health Scanner"
        )
        if "Chat" in st.session_state.app_mode:
            st.session_state.app_mode = "Chat"
        else:
            st.session_state.app_mode = "Livestock"
        st.divider()
    
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

# ════════════════════════════════════════════════════════════════
# ONBOARDING / REGISTRATION FLOW
# ════════════════════════════════════════════════════════════════

# --- Data: Maharashtra Region Crops ---
MAHARASHTRA_CROPS = {
    "Nashik": {
        "common": ["Grapes", "Onion", "Pomegranate", "Tomato", "Wheat", "Bajra", "Sugarcane", "Maize"],
        "rare": ["Saffron", "Avocado"],
        "climate": "Semi-arid, moderate rainfall",
        "soil": "Black soil, Medium-deep soils",
    },
    "Pune": {
        "common": ["Sugarcane", "Rice", "Wheat", "Jowar", "Bajra", "Vegetables", "Flowers (Rose, Marigold)", "Turmeric"],
        "rare": ["Avocado"],
        "climate": "Tropical wet & dry, moderate rainfall",
        "soil": "Red laterite, Black soil",
    },
    "Konkan": {
        "common": ["Rice", "Mango (Alphonso)", "Cashew", "Coconut", "Kokum", "Jackfruit", "Betel Nut", "Spices"],
        "rare": ["Coffee"],
        "climate": "Tropical, heavy monsoon",
        "soil": "Laterite, Coastal alluvial",
    },
    "Vidarbha": {
        "common": ["Cotton", "Soybean", "Orange (Nagpur)", "Jowar", "Tur Dal", "Wheat", "Chilli", "Sunflower"],
        "rare": ["Dragon Fruit"],
        "climate": "Hot semi-arid, moderate rainfall",
        "soil": "Black cotton soil, Deep clay",
    },
    "Ahmednagar": {
        "common": ["Sugarcane", "Onion", "Pomegranate", "Bajra", "Groundnut", "Sunflower", "Milk production", "Jowar"],
        "rare": ["Olive"],
        "climate": "Semi-arid, drought-prone areas",
        "soil": "Shallow to medium black soil",
    },
    "Mahabaleshwar": {
        "common": ["Strawberry", "Raspberry", "Mulberry", "Carrot", "Beetroot", "Peas", "Potato", "Leafy vegetables"],
        "rare": ["Blueberry"],
        "climate": "Hill station, high rainfall, cool climate",
        "soil": "Laterite, Rich humus forest soil",
    },
    "Solapur": {
        "common": ["Pomegranate", "Sugarcane", "Jowar", "Bajra", "Tur Dal", "Grapes", "Onion", "Groundnut"],
        "rare": ["Pistachio"],
        "climate": "Hot semi-arid, low rainfall, drought-prone",
        "soil": "Shallow black soil, Rocky terrain",
    },
    "Satara": {
        "common": ["Sugarcane", "Rice", "Turmeric", "Strawberry", "Potato", "Wheat", "Jowar", "Vegetables"],
        "rare": ["Kiwi"],
        "climate": "Moderate, good rainfall in western parts",
        "soil": "Red laterite, Black soil",
    },
}

# --- Data: Rare Crop Disease Database ---
RARE_CROP_DISEASES = {
    "Saffron": [
        {
            "name": "Corm Rot (Fusarium oxysporum)",
            "symptoms": "Yellowing and wilting of leaves, soft brown rot on corms, foul smell from infected bulbs, stunted growth",
            "prevention": "Use certified disease-free corms, practice crop rotation (3-4 years), ensure well-drained soil, avoid waterlogging, treat corms with fungicide before planting",
            "treatment": "Remove and destroy infected corms immediately, apply Carbendazim (Bavistin) 2g/L as soil drench, improve drainage",
            "pesticides": "Carbendazim (Bavistin) 50% WP — 2g/L soil drench; Copper Oxychloride — 3g/L; Trichoderma viride — bio-agent, 5g/kg corm treatment"
        },
        {
            "name": "Leaf Blight (Rhizoctonia crocinum)",
            "symptoms": "Dark brown to black spots on leaves, leaves dry from tips downward, reduced flower production",
            "prevention": "Avoid overhead irrigation, maintain proper spacing, remove crop debris after harvest",
            "treatment": "Spray Mancozeb 75% WP at 2.5g/L at first symptom appearance, repeat every 10-14 days",
            "pesticides": "Mancozeb 75% WP — 2.5g/L foliar spray; Chlorothalonil — 2g/L; Hexaconazole 5% EC — 1ml/L"
        },
    ],
    "Avocado": [
        {
            "name": "Phytophthora Root Rot",
            "symptoms": "Wilting despite adequate water, small pale-green leaves, branch dieback from tips, dark brown/black roots, fruit drop",
            "prevention": "Plant in well-drained soil or raised beds, use Phytophthora-resistant rootstocks (Dusa, Latas), avoid overwatering, mulch with coarse organic material",
            "treatment": "Apply Metalaxyl (Ridomil Gold) as soil drench, inject trunk with Phosphonate (Aliette), improve drainage urgently",
            "pesticides": "Metalaxyl (Ridomil Gold) 4% GR — 25g per tree; Fosetyl-Al (Aliette) 80% WP — 2.5g/L; Potassium Phosphonate — trunk injection"
        },
        {
            "name": "Anthracnose (Colletotrichum gloeosporioides)",
            "symptoms": "Dark circular lesions on fruits, black spots on leaves, fruit turns black during ripening, post-harvest decay",
            "prevention": "Prune to improve air circulation, harvest at correct maturity, avoid fruit injury during picking, pre-harvest fungicide sprays",
            "treatment": "Spray Azoxystrobin or Copper Hydroxide at flowering and fruit-set stage, post-harvest hot water treatment (48°C, 20 min)",
            "pesticides": "Azoxystrobin (Amistar) 23% SC — 1ml/L; Copper Hydroxide — 2g/L; Carbendazim — 1g/L; Prochloraz — post-harvest dip"
        },
        {
            "name": "Cercospora Leaf Spot",
            "symptoms": "Angular brown spots on leaves with yellow halos, premature leaf drop, reduced photosynthesis",
            "prevention": "Maintain tree vigor with balanced fertilization, remove fallen leaves, ensure good air circulation",
            "treatment": "Apply Copper-based fungicides, spray Mancozeb at 2-week intervals during wet season",
            "pesticides": "Mancozeb 75% WP — 2.5g/L; Copper Oxychloride — 3g/L; Thiophanate-methyl — 1g/L"
        },
    ],
    "Coffee": [
        {
            "name": "Coffee Leaf Rust (Hemileia vastatrix)",
            "symptoms": "Orange-yellow powdery spots on leaf undersides, premature leaf fall, bare branches, 30-80% yield loss if untreated",
            "prevention": "Plant rust-resistant varieties (Sln.5, Sln.9), shade management, balanced nutrition (potash enhances resistance), avoid dense planting",
            "treatment": "Spray Bordeaux mixture (1%) pre-monsoon, apply systemic fungicides at first sign of infection",
            "pesticides": "Bordeaux Mixture 1% — preventive spray; Tridemorph (Calixin) 80% EC — 0.5ml/L (2 sprays: June & September); Hexaconazole 5% EC — 2ml/L; Propiconazole — 1ml/L"
        },
        {
            "name": "Coffee Berry Disease (Colletotrichum kahawae)",
            "symptoms": "Dark sunken lesions on green berries, mummified black berries on branches, premature berry drop",
            "prevention": "Maintain shade canopy, prune for airflow, remove mummified berries, apply preventive fungicide before flowering",
            "treatment": "Spray Copper Oxychloride at pea-berry stage, follow with Carbendazim sprays every 3 weeks",
            "pesticides": "Copper Oxychloride 50% WP — 3g/L; Carbendazim 50% WP — 1g/L; Chlorothalonil — 2g/L"
        },
    ],
    "Dragon Fruit": [
        {
            "name": "Stem Rot (Enterobacter cloacae / Fusarium)",
            "symptoms": "Water-soaked soft spots on stems, yellowing at base of stem segments, brown mushy rot spreading upward, collapse of stem",
            "prevention": "Avoid overhead irrigation, ensure proper drainage, don't injure stems during harvesting, use sterilized cutting tools",
            "treatment": "Cut and remove infected stem parts (2 inches below infection), apply Bordeaux paste on cut surface, drench with Copper Oxychloride",
            "pesticides": "Bordeaux Paste — wound dressing; Copper Oxychloride 50% WP — 3g/L drench; Metalaxyl + Mancozeb — 2.5g/L; Trichoderma — bio-agent soil application"
        },
        {
            "name": "Anthracnose (Colletotrichum sp.)",
            "symptoms": "Brown sunken spots on fruit surface, cracking of fruit skin, reduced shelf life, internal browning",
            "prevention": "Pre-harvest sprays starting at flowering, avoid fruit injury, harvest at right maturity, proper post-harvest handling",
            "treatment": "Spray Azoxystrobin at flowering, apply Mancozeb every 14 days during fruit development",
            "pesticides": "Azoxystrobin 23% SC — 1ml/L; Mancozeb 75% WP — 2.5g/L; Carbendazim — 1g/L"
        },
    ],
    "Olive": [
        {
            "name": "Olive Knot (Pseudomonas savastanoi)",
            "symptoms": "Rough, woody galls/knots on branches and twigs, reduced fruiting on affected branches, branch dieback",
            "prevention": "Prune during dry weather only, sterilize pruning tools between cuts, avoid wounding trees, use resistant varieties",
            "treatment": "Prune all knotted branches 10cm below galls, apply Copper spray after pruning and before monsoon",
            "pesticides": "Copper Hydroxide 77% WP — 2.5g/L; Bordeaux Mixture 1% — post-pruning spray; Streptomycin Sulphate — 0.5g/L (severe cases)"
        },
        {
            "name": "Olive Leaf Spot (Spilocaea oleaginea)",
            "symptoms": "Dark circular spots on upper leaf surface, yellow halo around spots, premature leaf fall, reduced oil content in fruits",
            "prevention": "Ensure good air circulation through pruning, avoid dense canopy, balanced fertilization",
            "treatment": "Spray Copper-based fungicides before monsoon and after harvest, apply Dodine in severe cases",
            "pesticides": "Copper Oxychloride — 3g/L; Dodine 65% WP — 1g/L; Mancozeb — 2.5g/L"
        },
    ],
    "Blueberry": [
        {
            "name": "Mummy Berry (Monilinia vaccinii-corymbosi)",
            "symptoms": "Flowers turn brown and wilt, infected berries shrivel and become hard 'mummies' that fall to ground, gray spore masses on infected parts",
            "prevention": "Remove mummified berries from soil, apply 2-inch mulch layer to bury mummies, maintain good air circulation, avoid overhead watering",
            "treatment": "Spray Propiconazole during bud break, apply Chlorothalonil at bloom stage, remove infected berries immediately",
            "pesticides": "Propiconazole 25% EC — 1ml/L at bud break; Chlorothalonil 75% WP — 2g/L at bloom; Azoxystrobin — 1ml/L; Captan — 2g/L"
        },
        {
            "name": "Botrytis Blight / Gray Mold",
            "symptoms": "Gray fuzzy mold on ripening berries, blossom blight, soft watery decay of fruit, rapid spread in wet/humid conditions",
            "prevention": "Harvest frequently at proper ripeness, improve airflow through pruning, avoid overhead irrigation, cold-chain within 2 hours of harvest",
            "treatment": "Spray Iprodione or Fenhexamid at 10% bloom and repeat at full bloom, rotate fungicide groups",
            "pesticides": "Iprodione (Rovral) 50% WP — 2g/L; Fenhexamid (Elevate) — 1.5g/L; Cyprodinil + Fludioxonil (Switch) — 0.8g/L"
        },
    ],
    "Pistachio": [
        {
            "name": "Alternaria Late Blight",
            "symptoms": "Black lesions on leaves and nuts in late summer, staining of nut shells, premature leaf fall, reduced nut quality",
            "prevention": "Maintain open canopy through pruning, proper irrigation to avoid tree stress, balanced nitrogen fertilization",
            "treatment": "Spray Azoxystrobin or Pyraclostrobin at shell hardening stage, repeat every 14 days during humid periods",
            "pesticides": "Azoxystrobin 23% SC — 1ml/L; Pyraclostrobin — 0.5ml/L; Copper Hydroxide — 2.5g/L; Chlorothalonil — 2g/L"
        },
        {
            "name": "Botryosphaeria Panicle & Shoot Blight",
            "symptoms": "Dark brown killed flower clusters, shoot dieback, cankers on branches, infected nut shells turn black",
            "prevention": "Remove dead wood and mummified nut clusters, avoid sprinkler irrigation wetting canopy, prune for air circulation",
            "treatment": "Prune and destroy blighted shoots, apply Thiophanate-methyl or Pyraclostrobin at bud swell",
            "pesticides": "Thiophanate-methyl 70% WP — 1g/L; Pyraclostrobin + Boscalid — 0.5g/L; Copper Hydroxide — early season spray"
        },
    ],
    "Kiwi": [
        {
            "name": "Bacterial Canker (Pseudomonas syringae pv. actinidiae — PSA)",
            "symptoms": "Reddish-brown cankers with white/orange ooze on trunks, leaf spots with yellow halos, wilting of shoots, rapid vine death in severe cases",
            "prevention": "Use PSA-resistant varieties (Gold3/Sungold), sterilize all tools, avoid pruning in wet weather, monitor regularly, quarantine new plants",
            "treatment": "Cut cankers 30cm below visible infection, apply Copper sprays 3-4 times during dormancy and at bud break, destroy severely infected vines",
            "pesticides": "Copper Hydroxide 77% WP — 2.5g/L (pre-bud break, monthly); Kasugamycin — 2ml/L; Streptomycin Sulphate — 0.5g/L (as permitted)"
        },
        {
            "name": "Botrytis Fruit Rot (Gray Mold)",
            "symptoms": "Soft watery rot at stem end of fruit, gray fluffy mold growth, rapid post-harvest decay, affects stored fruit",
            "prevention": "Avoid fruit injury during harvest, harvest at correct maturity (6.2% Brix), pre-harvest fungicide spray, cold storage at 0°C immediately",
            "treatment": "Spray Iprodione or Fenhexamid pre-harvest (2 weeks before picking), dip harvested fruit in Fludioxonil solution",
            "pesticides": "Iprodione (Rovral) — 2g/L pre-harvest; Fenhexamid (Elevate) — 1.5g/L; Fludioxonil — post-harvest dip; Cyprodinil + Fludioxonil — 0.8g/L"
        },
    ],
}

# --- Helper: Step indicator ---
def render_step_indicator(current):
    steps_html = ""
    labels = ["Register", "Crop Plan", "Rare Crops"]
    for i in range(1, 4):
        if i < current:
            cls = "done"
        elif i == current:
            cls = "active"
        else:
            cls = "inactive"
        steps_html += f'<div class="step-dot {cls}">{i}</div>'
    return f'<div class="onboard-step">{steps_html}</div>'

# ════════════════════════════════════════════════════════
# MAIN APP GATE: Show onboarding OR main app
# ════════════════════════════════════════════════════════
if not st.session_state.farmer_registered:
    # ── ONBOARDING FLOW ──
    step = st.session_state.onboarding_step

    # ── STEP 1: FARMER REGISTRATION ──
    if step == 1:
        st.markdown(render_step_indicator(1), unsafe_allow_html=True)
        st.markdown("""
        <div class="onboard-container">
            <div class="onboard-title">🌾 Welcome to KrishiSakhiAI</div>
            <div class="onboard-subtitle">Let's set up your farmer profile to personalize your experience</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("registration_form"):
            st.markdown("### 👤 Farmer Registration")

            reg_c1, reg_c2 = st.columns(2)
            with reg_c1:
                farmer_name = st.text_input("👨‍🌾 Full Name *", placeholder="e.g. Rajesh Patil")
                farmer_phone = st.text_input("📱 Phone Number *", placeholder="e.g. 9876543210")
                farmer_land = st.number_input("🏞️ Land Size (Acres)", 0.1, 500.0, 5.0, 0.5,
                                               help="Total farming land in acres")
            with reg_c2:
                water_source = st.selectbox("💧 Primary Water Source", [
                    "🏔️ Well / Borewell",
                    "🏞️ Canal",
                    "🌊 River / Stream",
                    "🌧️ Rainfed only",
                    "🏗️ Farm Pond",
                    "🚰 Drip / Sprinkler System",
                    "🪣 Tank / Reservoir",
                ])
                current_crop = st.text_input("🌱 Current Crop(s) Growing",
                                              placeholder="e.g. Sugarcane, Onion, Wheat")
                farmer_region = st.selectbox("📍 Region (Maharashtra)", list(MAHARASHTRA_CROPS.keys()))

            submitted = st.form_submit_button("✅ Continue to Crop Planning →", use_container_width=True, type="primary")
            if submitted:
                if not farmer_name or not farmer_phone:
                    st.error("❌ Please fill in your Name and Phone Number")
                elif len(farmer_phone) < 10:
                    st.error("❌ Please enter a valid 10-digit phone number")
                else:
                    st.session_state.farmer_profile = {
                        'name': farmer_name,
                        'phone': farmer_phone,
                        'land_size': farmer_land,
                        'water_source': water_source,
                        'current_crop': current_crop,
                        'region': farmer_region,
                    }
                    st.session_state.selected_region = farmer_region
                    st.session_state.onboarding_step = 2
                    st.rerun()

    # ── STEP 2: CROP PLANNING ──
    elif step == 2:
        st.markdown(render_step_indicator(2), unsafe_allow_html=True)
        region = st.session_state.selected_region
        region_data = MAHARASHTRA_CROPS[region]
        profile = st.session_state.farmer_profile

        st.markdown(f"""
        <div class="crop-hero">
            <h3>🌾 Crop Planning — {region}</h3>
            <p>Welcome, <strong>{profile['name']}</strong>! Here are recommended crops for your region.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="region-card">
            <h4>📍 {region} Region Profile</h4>
            <p style="color:#cbd5e1; margin:0.3rem 0;"><strong>Climate:</strong> {region_data['climate']}</p>
            <p style="color:#cbd5e1; margin:0.3rem 0;"><strong>Soil Type:</strong> {region_data['soil']}</p>
            <p style="color:#cbd5e1; margin:0.3rem 0;"><strong>Your Land:</strong> {profile['land_size']} acres | <strong>Water:</strong> {profile['water_source']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🌱 Recommended Crops for Your Region")
        crop_cols = st.columns(4)
        for idx, crop in enumerate(region_data['common']):
            with crop_cols[idx % 4]:
                st.markdown(f"""
                <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                     border-radius:12px; padding:1rem; text-align:center; margin-bottom:0.8rem;">
                    <div style="font-size:1.1rem; font-weight:600; color:#10b981;">🌿 {crop}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown(f"### ✨ Rare / Exotic Crop Opportunity")
        st.markdown(f"Your region **{region}** has potential for growing these rare high-value crops:")
        for rare in region_data['rare']:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(251,191,36,0.1),rgba(245,158,11,0.05));
                 border:1px solid rgba(251,191,36,0.4); border-radius:14px; padding:1.2rem; margin:0.8rem 0;">
                <div style="font-size:1.3rem; font-weight:700; color:#fbbf24;">🌟 {rare}</div>
                <div style="color:#cbd5e1; font-size:0.9rem; margin-top:0.3rem;">
                    High-value crop rarely grown in {region}. Explore detailed disease management in the next step.
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back to Registration", use_container_width=True):
                st.session_state.onboarding_step = 1
                st.rerun()
        with col_next:
            if st.button("Explore Rare Crops & Diseases →", use_container_width=True, type="primary"):
                st.session_state.onboarding_step = 3
                st.rerun()

    # ── STEP 3: RARE CROP DISEASE EXPLORER ──
    elif step == 3:
        st.markdown(render_step_indicator(3), unsafe_allow_html=True)
        region = st.session_state.selected_region
        region_data = MAHARASHTRA_CROPS[region]
        profile = st.session_state.farmer_profile

        st.markdown(f"""
        <div class="crop-hero">
            <h3>🔬 Rare Crop Disease Guide — {region}</h3>
            <p>Complete disease management for high-value crops in your region</p>
        </div>
        """, unsafe_allow_html=True)

        for rare_crop in region_data['rare']:
            st.markdown(f"## 🌟 {rare_crop}")
            diseases = RARE_CROP_DISEASES.get(rare_crop, [])

            if not diseases:
                st.info(f"Disease data for {rare_crop} is being compiled.")
                continue

            for disease in diseases:
                card_html = f"""<div class="disease-card"><h5>🦠 {disease['name']}</h5><div class="disease-section symptoms"><strong>⚠️ Symptoms:</strong><br/>{disease['symptoms']}</div><div class="disease-section prevention"><strong>🛡️ Preventive Measures:</strong><br/>{disease['prevention']}</div><div class="disease-section treatment"><strong>💊 Treatment:</strong><br/>{disease['treatment']}</div><div class="disease-section pesticide"><strong>🧪 Recommended Pesticides / Fungicides:</strong><br/>{disease['pesticides']}</div></div>"""
                st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("")

        st.markdown("---")
        col_back3, col_finish = st.columns(2)
        with col_back3:
            if st.button("← Back to Crop Planning", use_container_width=True):
                st.session_state.onboarding_step = 2
                st.rerun()
        with col_finish:
            if st.button("✅ Complete Setup & Enter KrishiSakhiAI", use_container_width=True, type="primary"):
                st.session_state.farmer_registered = True
                st.session_state.onboarding_step = 4
                st.rerun()

else:
    # ════════════════════════════════════════════════════════════════
    # MAIN APP (only shown after onboarding is complete)
    # ════════════════════════════════════════════════════════════════

    # Show farmer greeting in sidebar
    with st.sidebar:
        if st.session_state.farmer_profile:
            fp = st.session_state.farmer_profile
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                 border-radius:12px; padding:0.8rem; margin-bottom:1rem; text-align:center;">
                <div style="font-weight:700; color:#10b981; font-size:1rem;">👨‍🌾 {fp['name']}</div>
                <div style="color:#94a3b8; font-size:0.8rem;">📍 {fp['region']} • {fp['land_size']} acres</div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🔄 Reset Profile", use_container_width=True):
            st.session_state.farmer_registered = False
            st.session_state.onboarding_step = 1
            st.session_state.farmer_profile = {}
            st.rerun()

    if st.session_state.app_mode == "Chat":
        # ════════════════════════════════════════════════════════════════
        # CHAT ASSISTANT MODE
        # ════════════════════════════════════════════════════════════════

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
                    <div class="topic-button">🐄 Livestock Health</div>
                </div>
                <p class="welcome-message" style="margin-top: 2rem; font-size: 1.1rem;">
                    Let's start! Type your question below, or switch to <strong>🐄 Livestock Health Scanner</strong> in the sidebar.
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

        # Footer (Chat mode)
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #64748b;">
            <p><strong>KrishiSakhiAI</strong> - Empowering farmers with AI-powered agricultural insights</p>
            <p style="font-size: 0.9rem;">Built with ❤️ for the farming community | Powered by Ollama & Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ════════════════════════════════════════════════════════════════
        # LIVESTOCK HEALTH SCANNER MODE
        # ════════════════════════════════════════════════════════════════

        livestock_models = st.session_state.livestock_models
        if not livestock_models or len(livestock_models) < 4:
            st.error("⚠️ Livestock models not found. Please run `python train_livestock_models.py` first.")
        else:
            # Hero
            st.markdown("""
            <div class="livestock-hero">
                <h2>🐄 Livestock Biosecurity Hub</h2>
                <p>AI diagnostics • Vaccination tracker • Diet manager • Farm inputs • Emergency vet finder</p>
            </div>
            """, unsafe_allow_html=True)

            # ═══ Sub-tabs ═══
            ls_tab1, ls_tab2, ls_tab3, ls_tab4, ls_tab5 = st.tabs([
                "🔬 Health Scanner",
                "💉 Vaccination & Medicine",
                "🥬 Feed & Diet",
                "🚜 Irrigation & Farm Inputs",
                "🚑 Emergency Vet Finder",
            ])

            # ──────────────────────────────────────
            # TAB 1: HEALTH SCANNER
            # ──────────────────────────────────────
            with ls_tab1:
                input_method = st.radio(
                    "Choose Input Method",
                    ["📝 Manual Sensor Input", "📁 Upload CSV File"],
                    horizontal=True,
                    help="Enter readings manually or upload CSV",
                    key="health_input_method"
                )

                if "Manual" in input_method:
                    st.markdown('<div class="input-section-label">🌡️ Physiological Sensors</div>', unsafe_allow_html=True)

                    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                    with col_s1:
                        body_temp = st.number_input("Body Temp (°C)", 35.0, 43.0, 38.8, 0.1, help="Normal: 38.3–39.4°C")
                    with col_s2:
                        heart_rate = st.number_input("Heart Rate (bpm)", 20, 150, 60, 1, help="Normal: 40–80 bpm")
                    with col_s3:
                        respiratory_rate = st.number_input("Resp. Rate (/min)", 5, 100, 22, 1, help="Normal: 15–30/min")
                    with col_s4:
                        activity_level = st.number_input("Activity (0-100)", 0, 100, 68, 1, help="Normal: 50–85")
                    with col_s5:
                        milk_yield = st.number_input("Milk Yield (L/day)", 0.0, 60.0, 20.0, 0.5, help="Normal: 12–30 L")

                    st.markdown('<div class="input-section-label">🔄 Activity & Intake Sensors</div>', unsafe_allow_html=True)

                    col_s6, col_s7, col_s8, col_s9, col_s10 = st.columns(5)
                    with col_s6:
                        rumination_min = st.number_input("Rumination (min/day)", 0, 700, 450, 10, help="Normal: 380–520 min")
                    with col_s7:
                        feed_intake = st.number_input("Feed Intake (kg/day)", 0.0, 50.0, 22.0, 0.5, help="Normal: 18–28 kg")
                    with col_s8:
                        water_intake = st.number_input("Water Intake (L/day)", 0.0, 150.0, 55.0, 1.0, help="Normal: 40–80 L")
                    with col_s9:
                        lying_time = st.number_input("Lying Time (hrs/day)", 0.0, 24.0, 12.0, 0.5, help="Normal: 10–14 hrs")
                    with col_s10:
                        steps_count = st.number_input("Steps (/day)", 0, 10000, 3500, 100, help="Normal: 2000–5000")

                    st.markdown('<div class="input-section-label">🦿 Gait & Mobility (from Camera/CV)</div>', unsafe_allow_html=True)

                    col_g1, col_g2, col_g3 = st.columns(3)
                    with col_g1:
                        gait_score = st.number_input("Gait Score (1-5)", 1.0, 5.0, 1.2, 0.1, help="1=Normal, 5=Severe Lameness")
                    with col_g2:
                        stance_symmetry = st.number_input("Stance Symmetry (0-1)", 0.0, 1.0, 0.95, 0.01, help="1.0 = perfect symmetry")
                    with col_g3:
                        stride_length = st.number_input("Stride Length (m)", 0.1, 2.5, 1.5, 0.05, help="Normal: 1.4–1.7m")

                    st.markdown('<div class="input-section-label">🌤️ Environment</div>', unsafe_allow_html=True)

                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        ambient_temp = st.number_input("Ambient Temp (°C)", 10.0, 50.0, 30.0, 0.5)
                    with col_e2:
                        humidity_pct = st.number_input("Humidity (%)", 10.0, 100.0, 65.0, 1.0)

                    thi_index = round(0.8 * ambient_temp + humidity_pct / 100 * (ambient_temp - 14.4) + 46.4, 1)

                    reading = {
                        'body_temp': body_temp, 'heart_rate': heart_rate, 'respiratory_rate': respiratory_rate,
                        'activity_level': activity_level, 'rumination_min': rumination_min,
                        'feed_intake': feed_intake, 'water_intake': water_intake, 'milk_yield': milk_yield,
                        'lying_time': lying_time, 'steps_count': steps_count,
                        'gait_score': gait_score, 'stance_symmetry': stance_symmetry, 'stride_length': stride_length,
                        'ambient_temp': ambient_temp, 'humidity_pct': humidity_pct, 'thi_index': thi_index,
                    }

                    st.markdown("")
                    if st.button("🔬 Run Biosecurity Scan", use_container_width=True, type="primary"):
                        with st.spinner("🔄 Running ML diagnostics..."):
                            health_result = livestock_models['health_predictor'].predict(reading)
                            anomaly_result = livestock_models['anomaly_detector'].predict(reading)
                            gait_result = livestock_models['gait_predictor'].predict(reading)
                            disease_result = livestock_models['disease_forecaster'].predict(reading)
                            gait_analyzer = GaitAnalyzer()
                            gait_cv = gait_analyzer.analyze_gait(reading)
                            behavior_analyzer = BehaviorAnalyzer()
                            behavior = behavior_analyzer.analyze_behavior(reading)

                            st.session_state.livestock_result = {
                                'health': health_result, 'anomaly': anomaly_result,
                                'gait': gait_result, 'disease': disease_result,
                                'gait_cv': gait_cv, 'behavior': behavior, 'reading': reading,
                            }

                    if st.session_state.livestock_result:
                        res = st.session_state.livestock_result
                        hr = res['health']; ar = res['anomaly']; gr = res['gait']
                        dr = res['disease']; gcv = res['gait_cv']; bh = res['behavior']
                        st.markdown("---")

                        if hr['status'] == 2:
                            st.markdown(f'<div class="alert-banner alert-banner-critical">🚨 <strong>CRITICAL:</strong> Immediate vet attention needed. Confidence: {hr["confidence"]*100:.0f}%. Condition: {dr["predicted_disease"].replace("_"," ")}.</div>', unsafe_allow_html=True)
                        elif hr['status'] == 1:
                            st.markdown(f'<div class="alert-banner alert-banner-warning">⚠️ <strong>AT-RISK:</strong> Early signs detected. Confidence: {hr["confidence"]*100:.0f}%. Monitor closely.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="alert-banner alert-banner-healthy">✅ <strong>HEALTHY:</strong> All parameters normal. Confidence: {hr["confidence"]*100:.0f}%.</div>', unsafe_allow_html=True)

                        sc = {0: '#10b981', 1: '#f97316', 2: '#ef4444'}
                        scc = {0: 'result-card-green', 1: 'result-card-orange', 2: 'result-card-red'}

                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                        with col_r1:
                            st.markdown(f'<div class="result-card {scc[hr["status"]]}"><div class="result-title">🏥 Health</div><div class="result-value" style="color:{sc[hr["status"]]};">{hr["status_label"]}</div><div class="result-subtitle">{hr["confidence"]*100:.1f}%</div></div>', unsafe_allow_html=True)
                        with col_r2:
                            dl = dr['predicted_disease'].replace('_', ' ')
                            dc = '#10b981' if dr['is_healthy'] else '#f97316'
                            dcc = 'result-card-green' if dr['is_healthy'] else 'result-card-orange'
                            st.markdown(f'<div class="result-card {dcc}"><div class="result-title">🦠 Disease</div><div class="result-value" style="color:{dc}; font-size:1.4rem;">{dl}</div><div class="result-subtitle">{dr["confidence"]*100:.1f}%</div></div>', unsafe_allow_html=True)
                        with col_r3:
                            gs = gr['gait_score']
                            gc = '#10b981' if gs < 2 else '#f97316' if gs < 3 else '#ef4444'
                            gcc = 'result-card-green' if gs < 2 else 'result-card-orange' if gs < 3 else 'result-card-red'
                            st.markdown(f'<div class="result-card {gcc}"><div class="result-title">🦿 Gait</div><div class="result-value" style="color:{gc};">{gs:.1f}/5</div><div class="result-subtitle">{gr["lameness_label"]}</div></div>', unsafe_allow_html=True)
                        with col_r4:
                            ac = '#ef4444' if ar['is_anomaly'] else '#10b981'
                            acc = 'result-card-red' if ar['is_anomaly'] else 'result-card-green'
                            al = 'ANOMALY' if ar['is_anomaly'] else 'NORMAL'
                            st.markdown(f'<div class="result-card {acc}"><div class="result-title">🔍 Anomaly</div><div class="result-value" style="color:{ac};">{al}</div><div class="result-subtitle">Score: {ar["anomaly_score"]:.4f}</div></div>', unsafe_allow_html=True)

                        with st.expander("📊 Health Probabilities"):
                            st.bar_chart(pd.DataFrame([hr['probabilities']]).T, color=['#10b981'])
                        with st.expander("🦠 Disease Probabilities"):
                            st.bar_chart(pd.DataFrame([dr['disease_probabilities']]).T, color=['#f97316'])
                        with st.expander("🦿 Gait Analysis"):
                            cv_score = gcv['locomotion_score']
                            gait_colors = ['#10b981', '#22c55e', '#eab308', '#f97316', '#ef4444']
                            seg = ""
                            for i in range(5):
                                op = '1.0' if (i+1) <= round(cv_score) else '0.15'
                                seg += f'<div class="gait-segment" style="background:{gait_colors[i]};opacity:{op};">{i+1}</div>'
                            st.markdown(f'<div style="text-align:center;"><div style="font-size:2.5rem;font-weight:800;color:{"#10b981" if cv_score<=2 else "#f97316" if cv_score<=3 else "#ef4444"};">{cv_score}/5 — {gcv["label"]}</div><div style="font-size:0.85rem;color:#94a3b8;">{gcv["description"]}</div><div class="gait-meter-bar">{seg}</div></div>', unsafe_allow_html=True)
                            for m, v in gcv['cv_metrics'].items():
                                st.markdown(f"- **{m.replace('_',' ').title()}:** {v}")
                        with st.expander("🧠 Behavior Analysis"):
                            st.markdown(f"**Pattern:** {bh['behavior_pattern'].replace('_',' ')}")
                            st.markdown(f"**Score:** {bh['behavior_health_score']}/100")
                            for m, v in bh['metrics'].items():
                                st.markdown(f"- **{m.replace('_',' ').title()}:** {v:.1f}/100")
                            if bh.get('alerts'):
                                for ba in bh['alerts']:
                                    st.warning(f"**{ba['type']}** ({ba['severity']}): {ba['message']}")

                else:
                    # CSV Upload
                    st.markdown('<div class="input-section-label">📁 Upload Sensor CSV</div>', unsafe_allow_html=True)
                    st.caption("Columns: `body_temp`, `heart_rate`, `respiratory_rate`, `activity_level`, `rumination_min`, `feed_intake`, `water_intake`, `milk_yield`, `lying_time`, `steps_count`, `gait_score`, `stance_symmetry`, `stride_length`, `ambient_temp`, `humidity_pct`, `thi_index`")
                    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="livestock_csv")
                    if uploaded_csv is not None:
                        try:
                            df_up = pd.read_csv(uploaded_csv)
                            st.success(f"✅ Loaded {len(df_up)} records")
                            st.dataframe(df_up.head(10), use_container_width=True)
                            if st.button("🔬 Run Batch Scan", use_container_width=True, type="primary"):
                                with st.spinner("🔄 Scanning..."):
                                    results = []
                                    for idx, row in df_up.iterrows():
                                        rd = row.to_dict()
                                        h = livestock_models['health_predictor'].predict(rd)
                                        d = livestock_models['disease_forecaster'].predict(rd)
                                        g = livestock_models['gait_predictor'].predict(rd)
                                        a = livestock_models['anomaly_detector'].predict(rd)
                                        results.append({'#': idx+1, 'ID': rd.get('animal_id', f'A-{idx+1}'), 'Health': h['status_label'], 'Conf': f"{h['confidence']*100:.0f}%", 'Disease': d['predicted_disease'].replace('_',' '), 'Gait': f"{g['gait_score']:.1f}", 'Anomaly': '⚠️' if a['is_anomaly'] else '✅'})
                                    st.dataframe(pd.DataFrame(results), use_container_width=True, height=400)
                        except Exception as e:
                            st.error(f"❌ {e}")

            # ──────────────────────────────────────
            # TAB 2: VACCINATION & MEDICINE
            # ──────────────────────────────────────
            with ls_tab2:
                st.markdown('<div class="input-section-label">💉 Vaccination & Medicine Tracker</div>', unsafe_allow_html=True)

                if 'vaccination_records' not in st.session_state:
                    st.session_state.vaccination_records = []

                with st.form("vaccination_form", clear_on_submit=True):
                    st.markdown("**Add New Vaccination / Medicine Entry**")
                    vc1, vc2, vc3 = st.columns(3)
                    with vc1:
                        vax_animal_id = st.text_input("Animal ID / Tag", placeholder="e.g. KS-0042", key="vax_aid")
                        vax_type = st.selectbox("Record Type", [
                            "Vaccination", "Deworming", "Antibiotic", "Anti-Inflammatory",
                            "Vitamin Supplement", "Mineral Supplement", "Hormonal Treatment", "Other Medicine"
                        ])
                    with vc2:
                        vax_name = st.text_input("Vaccine / Medicine Name", placeholder="e.g. FMD Vaccine, Ivermectin")
                        vax_dose = st.text_input("Dosage", placeholder="e.g. 5 mL intramuscular")
                    with vc3:
                        vax_date = st.date_input("Date Administered", key="vax_d1")
                        vax_next = st.date_input("Next Due Date", key="vax_d2")
                        vax_vet = st.text_input("Administered By", placeholder="Dr. Sharma")

                    vax_notes = st.text_area("Notes", placeholder="Batch number, reactions, withdrawal period...", height=70)

                    if st.form_submit_button("💾 Save Record", use_container_width=True):
                        if vax_name:
                            st.session_state.vaccination_records.append({
                                'Animal ID': vax_animal_id or 'N/A', 'Type': vax_type, 'Name': vax_name,
                                'Dosage': vax_dose, 'Date': str(vax_date), 'Next Due': str(vax_next),
                                'Vet': vax_vet, 'Notes': vax_notes,
                            })
                            st.success(f"✅ {vax_type} record saved!")

                st.markdown('<div class="input-section-label">📋 Common Vaccination Schedule (India)</div>', unsafe_allow_html=True)
                schedule = [
                    {"Disease": "Foot & Mouth Disease (FMD)", "Vaccine": "FMD Vaccine", "When": "At 4 months, then every 6 months", "Route": "Subcutaneous"},
                    {"Disease": "Hemorrhagic Septicemia (HS)", "Vaccine": "HS Vaccine", "When": "Before monsoon, annually", "Route": "Subcutaneous"},
                    {"Disease": "Black Quarter (BQ)", "Vaccine": "BQ Vaccine", "When": "Before monsoon, annually", "Route": "Subcutaneous"},
                    {"Disease": "Brucellosis", "Vaccine": "Brucella S19 / RB51", "When": "Female calves 4-8 months (once)", "Route": "Subcutaneous"},
                    {"Disease": "Theileriosis", "Vaccine": "Raksha Vac T", "When": "At 2-3 months", "Route": "Intramuscular"},
                    {"Disease": "Anthrax", "Vaccine": "Anthrax Spore Vaccine", "When": "Annually in endemic areas", "Route": "Subcutaneous"},
                    {"Disease": "Internal Parasites", "Vaccine": "Albendazole / Fenbendazole", "When": "Every 3-4 months", "Route": "Oral"},
                    {"Disease": "External Parasites", "Vaccine": "Ivermectin / Deltamethrin", "When": "Every 2-3 months", "Route": "Pour-on / Injection"},
                ]
                st.dataframe(pd.DataFrame(schedule), use_container_width=True, hide_index=True)

                if st.session_state.vaccination_records:
                    st.markdown('<div class="input-section-label">📂 Saved Records</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(st.session_state.vaccination_records), use_container_width=True, hide_index=True)
                    if st.button("🗑️ Clear All Records", key="clear_vax"):
                        st.session_state.vaccination_records = []
                        st.rerun()

            # ──────────────────────────────────────
            # TAB 3: FEED & DIET
            # ──────────────────────────────────────
            with ls_tab3:
                st.markdown('<div class="input-section-label">🥬 Animal Diet & Nutrition Manager</div>', unsafe_allow_html=True)

                if 'diet_records' not in st.session_state:
                    st.session_state.diet_records = []

                with st.form("diet_form", clear_on_submit=True):
                    st.markdown("**Log Daily Diet**")
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        diet_animal = st.text_input("Animal ID / Group", placeholder="e.g. KS-0042 or 'All Dairy Cows'", key="diet_aid")
                        diet_type = st.selectbox("Animal Type", [
                            "Dairy Cow", "Buffalo", "Calf (0-6 months)", "Heifer", "Bull",
                            "Pregnant Cow", "Dry Cow", "Goat", "Sheep"
                        ])
                    with dc2:
                        diet_date = st.date_input("Date", key="diet_d1")
                        diet_stage = st.selectbox("Lactation Stage", [
                            "Early Lactation (0-100 days)", "Mid Lactation (100-200 days)",
                            "Late Lactation (200-305 days)", "Dry Period", "Not Applicable"
                        ])

                    st.markdown("**🌾 Roughage**")
                    dr1, dr2, dr3, dr4 = st.columns(4)
                    with dr1:
                        green_fodder = st.number_input("Green Fodder (kg/day)", 0.0, 100.0, 25.0, 1.0, help="Napier, Maize, Berseem")
                    with dr2:
                        dry_fodder = st.number_input("Dry Fodder (kg/day)", 0.0, 30.0, 5.0, 0.5, help="Paddy straw, Wheat straw")
                    with dr3:
                        silage_amt = st.number_input("Silage (kg/day)", 0.0, 30.0, 0.0, 0.5, help="Maize silage")
                    with dr4:
                        hay_amt = st.number_input("Hay (kg/day)", 0.0, 15.0, 0.0, 0.5)

                    st.markdown("**🌽 Concentrates & Supplements**")
                    dr5, dr6, dr7, dr8 = st.columns(4)
                    with dr5:
                        conc_mix = st.number_input("Concentrate (kg/day)", 0.0, 20.0, 4.0, 0.5, help="Pellets, grain mix")
                    with dr6:
                        oil_cake = st.number_input("Oil Cake (kg/day)", 0.0, 5.0, 1.0, 0.25, help="Mustard cake, Soybean DOC")
                    with dr7:
                        mineral_mix = st.number_input("Mineral Mix (g/day)", 0.0, 200.0, 50.0, 5.0)
                    with dr8:
                        salt_amt = st.number_input("Salt (g/day)", 0.0, 100.0, 30.0, 5.0)

                    diet_water = st.number_input("💧 Water (L/day)", 0.0, 150.0, 60.0, 5.0, key="diet_water")
                    diet_notes = st.text_area("Special Supplements / Notes", placeholder="Bypass fat, probiotics, calcium supplement...", height=60)

                    if st.form_submit_button("💾 Save Diet Log", use_container_width=True):
                        st.session_state.diet_records.append({
                            'Animal': diet_animal or 'Herd', 'Type': diet_type, 'Date': str(diet_date),
                            'Stage': diet_stage, 'Green (kg)': green_fodder, 'Dry (kg)': dry_fodder,
                            'Silage (kg)': silage_amt, 'Conc. (kg)': conc_mix,
                            'Oil Cake (kg)': oil_cake, 'Mineral (g)': mineral_mix,
                            'Water (L)': diet_water, 'Notes': diet_notes,
                        })
                        st.success(f"✅ Diet log saved!")

                st.markdown('<div class="input-section-label">📖 Recommended Daily Diet (Adult Dairy Cow)</div>', unsafe_allow_html=True)
                ref_diet = [
                    {"Component": "🌿 Green Fodder", "Qty": "25–35 kg/day", "Examples": "Napier, CO-4, Berseem, Lucerne, Maize"},
                    {"Component": "🌾 Dry Fodder", "Qty": "5–8 kg/day", "Examples": "Paddy straw, Wheat straw, Ragi straw"},
                    {"Component": "🌽 Concentrate", "Qty": "1 kg per 2.5L milk + 1.5 kg", "Examples": "Grain mix, compound pellets"},
                    {"Component": "🫘 Oil Cake", "Qty": "1–2 kg/day", "Examples": "Groundnut, Mustard, Cotton seed cake"},
                    {"Component": "💊 Mineral Mix", "Qty": "50–80 g/day", "Examples": "Area-specific mineral mixture"},
                    {"Component": "🧂 Salt", "Qty": "25–35 g/day", "Examples": "Rock salt / iodised salt"},
                    {"Component": "💧 Water", "Qty": "50–80 L/day", "Examples": "Clean, fresh, ad-libitum"},
                ]
                st.dataframe(pd.DataFrame(ref_diet), use_container_width=True, hide_index=True)

                if st.session_state.diet_records:
                    st.markdown('<div class="input-section-label">📂 Diet Logs</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(st.session_state.diet_records), use_container_width=True, hide_index=True)

            # ──────────────────────────────────────
            # TAB 4: IRRIGATION & FARM INPUTS
            # ──────────────────────────────────────
            with ls_tab4:
                st.markdown('<div class="input-section-label">🚜 Farm Irrigation & Chemical Inputs</div>', unsafe_allow_html=True)

                if 'farm_input_records' not in st.session_state:
                    st.session_state.farm_input_records = []

                # Irrigation
                st.markdown("""
                <div class="result-card result-card-blue">
                    <div class="result-title">💧 Irrigation System</div>
                    <div class="result-subtitle" style="margin-top:0.3rem;">Configure irrigation for fodder cultivation</div>
                </div>
                """, unsafe_allow_html=True)

                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    irrigation_type = st.selectbox("Irrigation Method", [
                        "🚿 Drip Irrigation", "🌀 Sprinkler", "🌊 Flood / Surface",
                        "💧 Furrow", "🪣 Manual / Bucket",
                        "🏞️ Canal", "🔄 Center Pivot", "☁️ Rain-fed (No Irrigation)"
                    ])
                with ic2:
                    irrigation_source = st.selectbox("Water Source", [
                        "🏔️ Borewell / Tubewell", "🪨 Open Well", "🏞️ Canal / River",
                        "🌧️ Rainwater Harvesting", "🏗️ Farm Pond", "🚰 Municipal"
                    ])
                with ic3:
                    irrigation_freq = st.selectbox("Frequency", [
                        "Daily", "Every 2-3 days", "Weekly", "Bi-weekly",
                        "Sensor-based", "Seasonal only"
                    ])

                ic4, ic5 = st.columns(2)
                with ic4:
                    fodder_area = st.number_input("Fodder Area (acres)", 0.0, 100.0, 2.0, 0.5)
                with ic5:
                    water_usage = st.number_input("Est. Water (liters/day)", 0, 100000, 5000, 500)

                st.markdown("---")

                # Pesticide & Fertilizer
                st.markdown("""
                <div class="result-card result-card-orange">
                    <div class="result-title">🧪 Pesticide & Fertilizer Records</div>
                    <div class="result-subtitle" style="margin-top:0.3rem;">Track chemicals used on fodder fields — MRL compliance & biosecurity</div>
                </div>
                """, unsafe_allow_html=True)

                with st.form("farm_input_form", clear_on_submit=True):
                    fc1, fc2, fc3 = st.columns(3)
                    with fc1:
                        fi_type = st.selectbox("Input Type", [
                            "🧪 Chemical Pesticide", "🌿 Bio-Pesticide / Organic",
                            "🌱 Chemical Fertilizer", "🐄 Organic Manure / Compost",
                            "🦟 Insecticide", "🍄 Fungicide", "🌿 Herbicide",
                            "🌱 Growth Regulator", "🧫 Soil Amendment"
                        ])
                        fi_name = st.text_input("Product Name", placeholder="e.g. Urea, DAP, Neem oil")
                    with fc2:
                        fi_dosage = st.text_input("Dosage / Rate", placeholder="e.g. 50 kg/acre, 2 mL/L")
                        fi_field = st.text_input("Field / Plot", placeholder="e.g. Napier plot A")
                    with fc3:
                        fi_date = st.date_input("Application Date", key="fi_date")
                        fi_phi = st.number_input("Pre-Harvest Interval (days)", 0, 90, 14, 1,
                                                  help="Days to wait before using as animal feed")

                    fi_notes = st.text_area("Safety / MRL Notes", placeholder="Withdrawal period, precautions...", height=60)

                    if st.form_submit_button("💾 Save Farm Input", use_container_width=True):
                        if fi_name:
                            st.session_state.farm_input_records.append({
                                'Type': fi_type, 'Product': fi_name, 'Dosage': fi_dosage,
                                'Field': fi_field, 'Date': str(fi_date), 'PHI (days)': fi_phi,
                                'Notes': fi_notes, 'Irrigation': irrigation_type, 'Water Source': irrigation_source,
                            })
                            st.success(f"✅ Record saved!")

                if st.session_state.farm_input_records:
                    st.markdown('<div class="input-section-label">📂 Farm Input History</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(st.session_state.farm_input_records), use_container_width=True, hide_index=True)

                st.markdown('<div class="input-section-label">⚠️ MRL Quick Guidelines</div>', unsafe_allow_html=True)
                mrl = [
                    {"Chemical": "Chlorpyrifos", "Type": "Insecticide", "MRL (mg/kg)": "0.01–0.1", "PHI": "21-30 days", "Risk": "🔴 High"},
                    {"Chemical": "Imidacloprid", "Type": "Insecticide", "MRL (mg/kg)": "0.05–1.0", "PHI": "14-21 days", "Risk": "🟡 Medium"},
                    {"Chemical": "Neem Oil", "Type": "Bio-Pesticide", "MRL (mg/kg)": "Exempt", "PHI": "0-3 days", "Risk": "🟢 Low"},
                    {"Chemical": "Mancozeb", "Type": "Fungicide", "MRL (mg/kg)": "0.05–5.0", "PHI": "14-21 days", "Risk": "🟡 Medium"},
                    {"Chemical": "Glyphosate", "Type": "Herbicide", "MRL (mg/kg)": "0.1–2.0", "PHI": "30+ days", "Risk": "🔴 High"},
                    {"Chemical": "Trichoderma", "Type": "Bio-fungicide", "MRL (mg/kg)": "Exempt", "PHI": "0 days", "Risk": "🟢 Low"},
                ]
                st.dataframe(pd.DataFrame(mrl), use_container_width=True, hide_index=True)

            # ──────────────────────────────────────
            # TAB 5: EMERGENCY VET FINDER
            # ──────────────────────────────────────
            with ls_tab5:
                st.markdown('<div class="input-section-label">🚑 Emergency Veterinary Assistance</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="alert-banner alert-banner-critical">
                    🚨 <strong>In an emergency?</strong> Stay calm. Enter your location below to find the nearest vet.
                    <br>National Animal Helpline: <strong>1962</strong>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="result-card result-card-red">
                    <div class="result-title">📞 Emergency Helplines</div>
                    <div style="margin-top:0.5rem; line-height:2.2; font-size:0.95rem;">
                        <div>🇮🇳 <strong>National Animal Helpline:</strong> <span style="color:#ef4444; font-weight:700; font-size:1.3rem;">1962</span></div>
                        <div>🐄 <strong>NDDB:</strong> 1800-121-3456 (Toll-free)</div>
                        <div>🏥 <strong>State Animal Husbandry Dept:</strong> Contact district collector</div>
                        <div>💉 <strong>Disease Reporting:</strong> Report outbreaks to nearest vet hospital</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")
                st.markdown("**📍 Find Nearest Veterinary Services**")

                vc1, vc2 = st.columns(2)
                with vc1:
                    user_location = st.text_input("Your Location / Village / Pincode",
                                                   placeholder="e.g. Anand, Gujarat or 388001",
                                                   key="vet_location")
                with vc2:
                    emergency_type = st.selectbox("Emergency Type", [
                        "🚑 General Emergency", "🤰 Difficult Calving / Dystocia",
                        "🤕 Injury / Fracture", "🤧 Sudden Illness / Poisoning",
                        "🦠 Disease Outbreak", "🔬 Lab Testing",
                        "💉 Vaccination Camp", "🐄 Reproductive Issue / AI",
                    ])

                if user_location:
                    gmaps_url = f"https://www.google.com/maps/search/veterinary+hospital+near+{user_location.replace(' ', '+')}"
                    st.markdown(f"""
                    <div class="result-card result-card-green">
                        <div class="result-title">🗺️ Search Results</div>
                        <div style="margin-top:0.5rem;">
                            <p style="font-size:0.95rem; color:#cbd5e1;">
                                Searching for <strong>veterinary services</strong> near <strong>{user_location}</strong>
                            </p>
                            <a href="{gmaps_url}" target="_blank"
                               style="display:inline-block; margin-top:0.5rem; background:linear-gradient(135deg,#10b981,#059669);
                                      color:white; padding:0.7rem 1.5rem; border-radius:12px; text-decoration:none;
                                      font-weight:600; font-size:0.95rem;">
                                🗺️ Open in Google Maps →
                            </a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="input-section-label">🩺 Emergency First Aid Guide</div>', unsafe_allow_html=True)
                first_aid = [
                    {"Emergency": "🤰 Difficult Calving", "Action": "Call vet immediately. Don't pull forcefully. Keep cow calm.", "Time": "< 2 hrs"},
                    {"Emergency": "🤧 Bloat / Tympany", "Action": "Keep standing. Drench 100mL vegetable oil. Trocar if severe.", "Time": "< 1 hr"},
                    {"Emergency": "🐍 Snake Bite", "Action": "Note time. Tourniquet above bite. Rush to vet for anti-venom.", "Time": "< 30 min"},
                    {"Emergency": "🤧 Poisoning", "Action": "Identify poison. Activated charcoal (1-3 g/kg BW). Call vet.", "Time": "< 1 hr"},
                    {"Emergency": "🩸 Heavy Bleeding", "Action": "Apply pressure with clean cloth. Elevate. Keep calm.", "Time": "Immediate"},
                    {"Emergency": "🌡️ High Fever (>40.5°C)", "Action": "Cold water bath. Meloxicam injection. Electrolyte water.", "Time": "< 4 hrs"},
                    {"Emergency": "🦴 Fracture / Down", "Action": "Do NOT lift. Immobilize limb. Provide bedding. Call vet.", "Time": "< 6 hrs"},
                    {"Emergency": "⚡ Milk Fever", "Action": "IV Calcium Borogluconate (slow). Prop sternal. Monitor heart.", "Time": "< 1 hr"},
                ]
                st.dataframe(pd.DataFrame(first_aid), use_container_width=True, hide_index=True)

                st.markdown('<div class="input-section-label">🏥 Key Veterinary Resources</div>', unsafe_allow_html=True)
                resources = [
                    {"Service": "Government Vet Hospital", "How to Find": "District Collector / Block office", "Coverage": "All districts"},
                    {"Service": "NDDB AI Services", "How to Find": "1800-121-3456", "Coverage": "Major dairy regions"},
                    {"Service": "Mobile Vet Unit (MVU)", "How to Find": "State Animal Husbandry Dept.", "Coverage": "Select blocks"},
                    {"Service": "KVK (Krishi Vigyan Kendra)", "How to Find": "ICAR website → your district", "Coverage": "731 KVKs nationwide"},
                    {"Service": "Veterinary College", "How to Find": "Nearest DUVASU / state vet university", "Coverage": "State capitals"},
                    {"Service": "Private Vet Clinic", "How to Find": "Google Maps / JustDial", "Coverage": "Towns & cities"},
                ]
                st.dataframe(pd.DataFrame(resources), use_container_width=True, hide_index=True)

            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #64748b;">
                <p><strong>KrishiSakhiAI</strong> — Livestock Biosecurity Hub</p>
                <p style="font-size: 0.9rem;">Health Scanner • Vaccination • Diet • Farm Inputs • Emergency Vet | Built with ❤️ for Farmers</p>
            </div>
            """, unsafe_allow_html=True)
