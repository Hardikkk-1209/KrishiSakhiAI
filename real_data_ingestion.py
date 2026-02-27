# real_data_ingestion.py
import os
import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
import json
import PyPDF2
from sentence_transformers import SentenceTransformer
import re
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticChunker:
    """Handles semantic chunking of text using sentence similarity."""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Using a lightweight sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def semantic_chunk(self, text: str) -> List[str]:
        """Create semantic chunks based on sentence similarity."""
        sentences = self.split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]
        
        # Get sentence embeddings
        sentence_embeddings = self.sentence_model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Find the best breaking point using semantic similarity
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Create overlap by including some previous sentences
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
            i += 1
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on the overlap size."""
        overlap_text = ""
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_text += sentence
            else:
                break
        
        return overlap_sentences

class OllamaEmbedder:
    """Handles embedding generation using Ollama API."""
    
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback to zero vector if Ollama fails
            return [0.0] * 768  # Default embedding size
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding {i+1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

class DocumentProcessor:
    """Handles processing of PDF and CSV files."""
    
    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def process_csv(self, csv_path: Path) -> str:
        """Process CSV file and convert to text format."""
        try:
            df = pd.read_csv(csv_path)
            
            # Convert DataFrame to structured text
            text_parts = [f"CSV File: {csv_path.name}\n"]
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}\n")
            text_parts.append(f"Total rows: {len(df)}\n\n")
            
            # Add column descriptions and sample data
            for col in df.columns:
                text_parts.append(f"Column '{col}':\n")
                text_parts.append(f"  - Data type: {df[col].dtype}\n")
                text_parts.append(f"  - Non-null values: {df[col].count()}\n")
                
                if df[col].dtype in ['object', 'string']:
                    unique_vals = df[col].dropna().unique()[:5]
                    text_parts.append(f"  - Sample values: {', '.join(map(str, unique_vals))}\n")
                else:
                    text_parts.append(f"  - Min: {df[col].min()}, Max: {df[col].max()}\n")
                text_parts.append("\n")
            
            # Add sample rows as structured text
            text_parts.append("Sample data rows:\n")
            for idx, row in df.head(10).iterrows():
                text_parts.append(f"Row {idx + 1}:\n")
                for col, val in row.items():
                    text_parts.append(f"  {col}: {val}\n")
                text_parts.append("\n")
            
            return ''.join(text_parts)
        except Exception as e:
            logger.error(f"Error processing CSV {csv_path}: {e}")
            return ""
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all PDF and CSV documents from the source path."""
        documents = []
        
        if not self.source_path.exists():
            logger.error(f"Source path does not exist: {self.source_path}")
            return documents
        
        # Process PDF files
        for pdf_file in self.source_path.glob("*.pdf"):
            logger.info(f"Processing PDF: {pdf_file.name}")
            text = self.extract_pdf_text(pdf_file)
            if text.strip():
                documents.append({
                    'filename': pdf_file.name,
                    'file_type': 'pdf',
                    'content': text,
                    'path': str(pdf_file)
                })
        
        # Process CSV files
        for csv_file in self.source_path.glob("*.csv"):
            logger.info(f"Processing CSV: {csv_file.name}")
            text = self.process_csv(csv_file)
            if text.strip():
                documents.append({
                    'filename': csv_file.name,
                    'file_type': 'csv',
                    'content': text,
                    'path': str(csv_file)
                })
        
        return documents

class FAISSVectorStore:
    """FAISS vector store for document embeddings."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.documents = []
        self.chunks_metadata = []
    
    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        """Add document chunks and embeddings to the vector store."""
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self.documents.extend(chunks)
        self.chunks_metadata.extend(metadata)
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        scores, indices = self.index.search(query_array, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'content': self.documents[idx],
                    'metadata': self.chunks_metadata[idx]
                })
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """Save the FAISS index and metadata."""
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'chunks_metadata': self.chunks_metadata,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"Vector store saved to {index_path} and {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """Load the FAISS index and metadata."""
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.chunks_metadata = data['chunks_metadata']
            self.dimension = data['dimension']
        
        logger.info(f"Vector store loaded from {index_path} and {metadata_path}")

class DocumentIndexer:
    """Main class to orchestrate the indexing process."""
    
    def __init__(self, source_path: str, chunk_size: int = 2000, overlap: int = 200):
        self.source_path = source_path
        self.doc_processor = DocumentProcessor(source_path)
        self.chunker = SemanticChunker(chunk_size, overlap)
        self.embedder = OllamaEmbedder()
        self.vector_store = FAISSVectorStore()
    
    def create_index(self, output_dir: str = "vector_store"):
        """Create the complete vector store index."""
        logger.info("Starting document indexing process...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all documents
        documents = self.doc_processor.get_all_documents()
        if not documents:
            logger.error("No documents found to process!")
            return
        
        logger.info(f"Found {len(documents)} documents to process")
        
        all_chunks = []
        all_metadata = []
        
        # Process each document
        for doc in documents:
            logger.info(f"Processing {doc['filename']}")
            
            # Create semantic chunks
            chunks = self.chunker.semantic_chunk(doc['content'])
            logger.info(f"Created {len(chunks)} chunks from {doc['filename']}")
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                metadata = {
                    'filename': doc['filename'],
                    'file_type': doc['file_type'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'path': doc['path'],
                    'chunk_length': len(chunk)
                }
                all_chunks.append(chunk)
                all_metadata.append(metadata)
        
        logger.info(f"Total chunks to embed: {len(all_chunks)}")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.generate_embeddings_batch(all_chunks)
        
        # Add to vector store
        self.vector_store.add_documents(all_chunks, embeddings, all_metadata)
        
        # Save vector store
        index_path = output_path / "faiss_index.bin"
        metadata_path = output_path / "metadata.pkl"
        self.vector_store.save(str(index_path), str(metadata_path))
        
        logger.info("Indexing complete!")
        return self.vector_store
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search documents using a query."""
        query_embedding = self.embedder.generate_embedding(query)
        return self.vector_store.search(query_embedding, k)

def main():
    """Main function to run the indexing process."""
    # Configuration
    SOURCE_PATH = r"real_agricultural_data"
    CHUNK_SIZE = 2000
    OVERLAP = 200
    OUTPUT_DIR = "agricultural_vector_store"
    
    # Create indexer
    indexer = DocumentIndexer(SOURCE_PATH, CHUNK_SIZE, OVERLAP)
    
    try:
        # Create the vector store
        vector_store = indexer.create_index(OUTPUT_DIR)
        
        # Example search
        if vector_store and vector_store.index.ntotal > 0:
            print("\n" + "="*50)
            print("TESTING SEARCH FUNCTIONALITY")
            print("="*50)
            
            query = "agricultural crop yield data"
            results = indexer.search_documents(query, k=3)
            
            print(f"\nSearch query: '{query}'")
            print(f"Found {len(results)} results:")
            
            for result in results:
                print(f"\nRank {result['rank']}: Score {result['score']:.4f}")
                print(f"File: {result['metadata']['filename']}")
                print(f"Chunk {result['metadata']['chunk_index']+1}/{result['metadata']['total_chunks']}")
                print(f"Content preview: {result['content'][:200]}...")
                print("-" * 50)
    
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise

if __name__ == "__main__":
    main()