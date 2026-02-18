#!/usr/bin/env python3
"""
RAG Pipeline for Email Dataset

A complete Retrieval-Augmented Generation system built from scratch
for querying a dataset of 100 synthetic emails.

Usage:
    python rag_pipeline.py

This script will:
1. Load and process all email documents
2. Generate embeddings using sentence-transformers
3. Build a FAISS index for efficient retrieval
4. Start an interactive Q&A session
"""

import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any

# Import our custom modules
from document_processor import EmailProcessor
from embedding_engine import EmbeddingEngine
from retrieval_engine import RetrievalEngine
from generator import AnswerGenerator


class RAGPipeline:
    """Main RAG Pipeline class that orchestrates all components"""
    
    def __init__(self, emails_dir: str = "emails"):
        """
        Initialize the RAG pipeline
        
        Args:
            emails_dir: Directory containing email files
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.email_processor = EmailProcessor(emails_dir)
        self.embedding_engine = EmbeddingEngine()
        self.retrieval_engine = RetrievalEngine(self.embedding_engine)
        self.answer_generator = AnswerGenerator()
        
        # Pipeline state
        self.chunks = []
        self.is_initialized = False
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the complete pipeline by processing documents and building index
        
        Returns:
            Dictionary with initialization statistics
        """
        print("🚀 Initializing RAG Pipeline...")
        print("=" * 50)
        
        try:
            # Step 1: Load and process emails
            print("📧 Loading and processing emails...")
            emails = self.email_processor.load_emails()
            print(f"   Loaded {len(emails)} emails")
            
            # Step 2: Chunk documents
            print("✂️  Chunking documents...")
            self.chunks = self.email_processor.chunk_emails(emails)
            print(f"   Created {len(self.chunks)} chunks")
            
            # Step 3: Build retrieval index
            print("🔍 Building retrieval index...")
            self.retrieval_engine.build_index(self.chunks)
            
            # Step 4: Get system info
            embedding_info = self.embedding_engine.get_embedding_info()
            index_stats = self.retrieval_engine.get_index_stats()
            model_info = self.answer_generator.get_model_info()
            
            self.is_initialized = True
            
            print("✅ Pipeline initialized successfully!")
            print("=" * 50)
            
            return {
                'emails_loaded': len(emails),
                'chunks_created': len(self.chunks),
                'embedding_model': embedding_info['model_name'],
                'embedding_dim': embedding_info['embedding_dim'],
                'index_stats': index_stats,
                'generator_model': model_info['model_name']
            }
            
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user query and return an answer
        
        Args:
            question: User's question
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        print(f"\n🔎 Processing query: '{question}'")
        
        try:
            # Step 1: Retrieve relevant chunks
            print("📚 Retrieving relevant documents...")
            retrieved_results = self.retrieval_engine.search(question, k=top_k)
            
            if not retrieved_results:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': [],
                    'retrieval_count': 0
                }
            
            print(f"   Found {len(retrieved_results)} relevant chunks")
            
            # Step 2: Generate answer
            print("🤖 Generating answer...")
            response = self.answer_generator.generate_answer(question, retrieved_results)
            
            # Add query and retrieval info to response
            response['question'] = question
            response['retrieval_count'] = len(retrieved_results)
            
            print("✅ Answer generated successfully!")
            
            return response
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {
                'question': question,
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'error': str(e)
            }
    
    def interactive_session(self):
        """Start an interactive Q&A session"""
        if not self.is_initialized:
            print("❌ Pipeline not initialized. Please run initialize() first.")
            return
        
        print("\n🎯 RAG Pipeline Interactive Session")
        print("=" * 50)
        print("Type your questions about the emails below.")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("Type 'help' for available commands.")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n❓ Ask a question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if question.lower() == 'stats':
                    self._show_stats()
                    continue
                
                # Process the query
                response = self.query(question)
                
                # Display the answer
                print(f"\n💬 Answer:")
                print(response['answer'])
                
                # Show sources if available
                if response.get('sources'):
                    print(f"\n📖 Sources (similarity scores):")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"   {i}. {source['filename']} (score: {source['similarity_score']:.3f})")
                        print(f"      Preview: {source['content_preview']}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show available commands"""
        print("\n📋 Available Commands:")
        print("  help  - Show this help message")
        print("  stats - Show pipeline statistics")
        print("  quit  - Exit the interactive session")
        print("  exit  - Exit the interactive session")
        print("  q     - Exit the interactive session")
    
    def _show_stats(self):
        """Show pipeline statistics"""
        print(f"\n📊 Pipeline Statistics:")
        print(f"  Total chunks: {len(self.chunks)}")
        print(f"  Embedding model: {self.embedding_engine.model_name}")
        print(f"  Embedding dimension: {self.embedding_engine.embedding_dim}")
        print(f"  Generator model: {self.answer_generator.model_name}")
        
        index_stats = self.retrieval_engine.get_index_stats()
        print(f"  Index status: {index_stats['status']}")
        if index_stats['status'] == 'built':
            print(f"  Index type: {index_stats['index_type']}")


def main():
    """Main function to run the RAG pipeline"""
    try:
        # Initialize the pipeline
        pipeline = RAGPipeline()
        
        # Initialize all components
        init_stats = pipeline.initialize()
        
        # Display initialization summary
        print("\n📊 Initialization Summary:")
        print(f"  Emails processed: {init_stats['emails_loaded']}")
        print(f"  Chunks created: {init_stats['chunks_created']}")
        print(f"  Embedding model: {init_stats['embedding_model']}")
        print(f"  Generator model: {init_stats['generator_model']}")
        
        # Start interactive session
        pipeline.interactive_session()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
