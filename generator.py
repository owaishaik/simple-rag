import os
import google.generativeai as genai
from typing import List, Tuple, Dict, Any
from document_processor import EmailChunk


class AnswerGenerator:
    """Handles answer generation using Gemini API"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-pro"):
        """
        Initialize the answer generator
        
        Args:
            api_key: Gemini API key (if None, will try to load from environment)
            model_name: Gemini model to use
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        print(f"Initialized Gemini model: {model_name}")
    
    def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[Tuple[EmailChunk, float]], 
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """
        Generate an answer based on the query and retrieved context
        
        Args:
            query: User's question
            retrieved_chunks: List of (chunk, similarity_score) tuples
            max_context_length: Maximum length of context to include in prompt
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'context_used': False
            }
        
        # Prepare context from retrieved chunks
        context = self._prepare_context(retrieved_chunks, max_context_length)
        
        # Create the prompt
        prompt = self._create_prompt(query, context)
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Extract sources
            sources = [
                {
                    'filename': chunk.metadata['filename'],
                    'chunk_id': chunk.chunk_id,
                    'similarity_score': score,
                    'content_preview': chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                }
                for chunk, score in retrieved_chunks
            ]
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': True,
                'context_length': len(context),
                'num_sources': len(sources)
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error while generating the answer: {str(e)}",
                'sources': [],
                'context_used': False,
                'error': str(e)
            }
    
    def _prepare_context(
        self, 
        retrieved_chunks: List[Tuple[EmailChunk, float]], 
        max_length: int
    ) -> str:
        """
        Prepare context string from retrieved chunks
        
        Args:
            retrieved_chunks: List of (chunk, similarity_score) tuples
            max_length: Maximum length of context to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, (chunk, similarity) in enumerate(retrieved_chunks):
            chunk_text = f"[Source {i+1}] {chunk.content}"
            
            # Check if adding this chunk would exceed the limit
            if current_length + len(chunk_text) > max_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the language model
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Complete prompt string
        """
        prompt = f"""You are a helpful assistant that answers questions based on the provided email context. 

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain enough information to answer the question, say so clearly.
3. Be concise but thorough in your answer.
4. If multiple emails provide relevant information, synthesize the information.
5. Cite the sources by referring to [Source 1], [Source 2], etc. in your answer.
6. Do not make up information that is not present in the context.

ANSWER:"""
        
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'api_configured': bool(self.api_key),
            'provider': 'Google Generative AI'
        }
