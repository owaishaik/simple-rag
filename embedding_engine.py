import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from document_processor import EmailChunk


class EmbeddingEngine:
    """Handles embedding generation and management"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension by testing with a sample sentence
            sample_embedding = self.model.encode(["test"])
            self.embedding_dim = sample_embedding.shape[1]
            print(f"Loaded model: {self.model_name}")
            print(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")
    
    def embed_chunks(self, chunks: List[EmailChunk]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks
        
        Args:
            chunks: List of EmailChunk objects
            
        Returns:
            numpy array of embeddings with shape (len(chunks), embedding_dim)
        """
        if not chunks:
            return np.array([])
        
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches for better memory efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        
        # Normalize embeddings for better similarity search
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query string
            
        Returns:
            numpy array of embedding with shape (embedding_dim,)
        """
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_sequence_length': getattr(self.model, 'max_seq_length', None)
        }
