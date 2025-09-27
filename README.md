# KrishiSakhiAI 🌾

**Kerala's Smart Farming Companion** - An AI-powered agricultural advisor that provides instant, personalized guidance through advanced language models and semantic search, turning expert knowledge into simple, actionable farming tips.

## Features

- **🤖 AI-Powered Conversations**: Chat with agricultural experts powered by Ollama language models
- **📚 Knowledge Base Search**: Semantic search through agricultural documents using FAISS vector store
- **🌱 Comprehensive Coverage**: Crop management, pest control, seasonal farming, irrigation, sustainability, and farm economics
- **💬 Natural Language**: Supports both Malayalam and English for local farmers
- **🔍 Context-Aware Responses**: Retrieval-Augmented Generation (RAG) provides relevant, accurate information
- **📊 Multi-Format Support**: Processes PDF and CSV agricultural documents

## Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- Agricultural documents (PDFs/CSVs) for knowledge base

### Installation

1. Clone the repository:
```bash
git clone "https://github.com/Hardikkk-1209/KrishiSakhiAI"
cd KrishiSakhiAI
```

2. Install dependencies:
```bash
pip install streamlit requests numpy faiss-cpu pandas pathlib sentence-transformers scikit-learn PyPDF2
```

3. Install and start Ollama:
```bash
# Install Ollama from https://ollama.ai
ollama serve

# Pull required models
ollama pull llama3
ollama pull nomic-embed-text
```

### Setup Knowledge Base

1. Place your agricultural documents (PDFs and CSVs) in the `real_agricultural_data/` directory

2. Run the document indexer:
```bash
python real_data_ingestion.py
```

This creates a vector store in `agricultural_vector_store/` containing:
- FAISS index (`faiss_index.bin`)
- Document metadata (`metadata.pkl`)

### Launch Application

```bash
streamlit run frontend.py
```

Visit `http://localhost:8501` to access KrishiSakhiAI.

## Architecture

### Core Components

- **`frontend.py`**: Streamlit web interface with dark agricultural theme
- **`real_data_ingestion.py`**: Document processing and vector store creation
- **Vector Store**: FAISS-based semantic search system
- **Ollama Integration**: Local LLM inference and embedding generation

### Data Flow

1. **Document Processing**: PDFs and CSVs are processed and chunked semantically
2. **Embedding Generation**: Text chunks are converted to vector embeddings via Ollama
3. **Vector Storage**: Embeddings stored in FAISS index for fast similarity search
4. **Query Processing**: User questions trigger semantic search + LLM generation
5. **Response Generation**: Context-enriched prompts generate relevant agricultural advice

## Configuration

### Model Settings
- **Default LLM**: Configurable via sidebar (llama2, mistral, etc.)
- **Embedding Model**: `nomic-embed-text`
- **Temperature**: Adjustable creativity level (0.0-2.0)

### Document Processing
- **Chunk Size**: 2000 characters (configurable)
- **Overlap**: 200 characters for context continuity
- **Search Results**: 1-5 context documents per query

## Usage Examples

**Sample Questions:**
- "What crops should I plant in Kharif season?"
- "How to control aphids on my tomato crop?"
- "Best fertilizer schedule for wheat?"
- "Water management techniques for rice farming"

## Troubleshooting

### Common Issues

**Ollama Connection Failed:**
- Ensure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`

**Knowledge Base Not Found:**
- Run `python real_data_ingestion.py` first
- Verify documents exist in `real_agricultural_data/`

**Slow Response Times:**
- Reduce search results count in sidebar
- Use smaller language models
- Ensure sufficient system resources

## File Structure

```
KrishiSakhiAI/
├── frontend.py                    # Main Streamlit application
├── real_data_ingestion.py        # Document indexing system
├── real_agricultural_data/       # Source documents (PDFs, CSVs)
├── agricultural_vector_store/     # Generated vector store
│   ├── faiss_index.bin           # FAISS similarity index
│   └── metadata.pkl              # Document metadata
└── README.md                     # This file
```

## Technology Stack

- **Frontend**: Streamlit
- **Vector Search**: FAISS
- **Language Models**: Ollama (llama3, mistral, etc.)
- **Embeddings**: nomic-embed-text via Ollama
- **Document Processing**: PyPDF2, pandas
- **Semantic Chunking**: sentence-transformers

## Contributing

1. Fork the repository
2. Create feature branch
3. Add agricultural knowledge or improve functionality
4. Test with sample queries
5. Submit pull request

## License

Open source - designed to support farmers and agricultural communities.

---

**KrishiSakhiAI** - Empowering farmers with AI-driven agricultural intelligence.
