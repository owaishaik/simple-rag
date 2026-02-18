import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from document_processor import EmailChunk
from embedding_engine import EmbeddingEngine


class RetrievalEngine:
    """Handles document retrieval using FAISS"""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        """
        Initialize the retrieval engine
        
        Args:
            embedding_engine: Instance of EmbeddingEngine
        """
        self.embedding_engine = embedding_engine
        self.index = None
        self.chunks = []
        self.is_built = False
    
    def build_index(self, chunks: List[EmailChunk]):
        """
        Build FAISS index from document chunks
        
        Args:
            chunks: List of EmailChunk objects
        """
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        print("Generating embeddings for chunks...")
        embeddings = self.embedding_engine.embed_chunks(chunks)
        
        # Store chunks for later retrieval
        self.chunks = chunks
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        
        # Use Inner Product (IP) since we normalize embeddings
        # This is equivalent to cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        self.is_built = True
        print(f"Built FAISS index with {len(chunks)} chunks")
        print(f"Index dimension: {embedding_dim}")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[EmailChunk, float]]:
        """
        Search for relevant chunks given a query
        
        Args:
            query: Query string
            k: Number of top results to return
            
        Returns:
            List of tuples containing (chunk, similarity_score)
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Call build_index() first.")
        
        if k <= 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        k = min(k, len(self.chunks))  # Ensure k doesn't exceed number of chunks
        similarities, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i in range(len(indices[0])):
            chunk_idx = indices[0][i]
            similarity = similarities[0][i]
            
            if chunk_idx >= 0:  # FAISS returns -1 for invalid indices
                chunk = self.chunks[chunk_idx]
                results.append((chunk, float(similarity)))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> EmailChunk:
        """
        Retrieve a specific chunk by its ID
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            EmailChunk object if found, None otherwise
        """
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if not self.is_built:
            return {'status': 'not_built'}
        
        return {
            'status': 'built',
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embedding_engine.embedding_dim,
            'index_type': 'IndexFlatIP (Inner Product)',
            'is_trained': self.index.is_trained if self.index else False
        }
    
    def save_index(self, filepath: str):
        """
        Save FAISS index to file
        
        Args:
            filepath: Path to save the index
        """
        if not self.is_built:
            raise RuntimeError("No index to save")
        
        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str, chunks: List[EmailChunk]):
        """
        Load FAISS index from file
        
        Args:
            filepath: Path to the saved index
            chunks: List of chunks corresponding to the index
        """
        try:
            self.index = faiss.read_index(filepath)
            self.chunks = chunks
            self.is_built = True
            print(f"Index loaded from {filepath}")
            print(f"Loaded {len(chunks)} chunks")
        except Exception as e:
            raise RuntimeError(f"Failed to load index from {filepath}: {e}")
