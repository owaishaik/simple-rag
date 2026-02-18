# Mini RAG System

## Overview

A complete Retrieval-Augmented Generation (RAG) pipeline built from scratch for querying a dataset of 100 synthetic emails. This implementation demonstrates the core components of a RAG system without using end-to-end frameworks like LangChain or LlamaIndex.

## Architecture

The system consists of four main components:

1. **Document Processor** (`document_processor.py`) - Loads and chunks email documents
2. **Embedding Engine** (`embedding_engine.py`) - Generates embeddings using sentence-transformers
3. **Retrieval Engine** (`retrieval_engine.py`) - Performs similarity search using FAISS
4. **Answer Generator** (`generator.py`) - Generates responses using Gemini API

## Dataset

A dataset of 100 synthetic emails is provided in the `emails/` directory. Each email contains:
- Subject line
- Sender and receiver information (from a pool of 200 unique people)
- Body content (100+ words) with diverse topics including:
  - Project updates
  - Meeting requests
  - Budget approvals
  - Technical issues
  - Client feedback
  - Team announcements
  - Deadline extensions
  - Training opportunities
  - Vendor proposals
  - Performance reviews

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key in the `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### For CPU-only systems (recommended):
```bash
source venv/bin/activate
CUDA_VISIBLE_DEVICES="" python rag_pipeline.py
```

### For systems with CUDA support:
```bash
source venv/bin/activate
python rag_pipeline.py
```

The system will:
1. Load and process all email documents
2. Generate embeddings for document chunks
3. Build a FAISS index for efficient retrieval
4. Start an interactive Q&A session

### Interactive Commands

- **Ask questions**: Simply type your question about the emails
- **help**: Show available commands
- **stats**: Display pipeline statistics
- **quit/exit/q**: Exit the session

### Example Queries

- "What training opportunities are mentioned?"
- "Are there any budget approvals?"
- "Tell me about project updates"
- "What meeting requests were sent?"
- "Any technical issues reported?"

## Design Choices and Tradeoffs

### Chunking Strategy
- **Approach**: Multi-level chunking (subject, body, full email)
- **Chunk Size**: 500 characters with 100-character overlap
- **Tradeoff**: Smaller chunks provide better granularity but may lose context; larger chunks preserve context but reduce precision

### Embedding Model
- **Choice**: `sentence-transformers/all-MiniLM-L6-v2`
- **Advantages**: Fast inference, good quality for English text, small footprint (384 dimensions)
- **Tradeoff**: Less powerful than larger models like BERT-large, but sufficient for this dataset

### Retrieval Method
- **Technology**: FAISS IndexFlatIP (Inner Product)
- **Advantages**: Exact similarity search, fast for this scale, memory efficient
- **Tradeoff**: Not suitable for millions of documents (would need approximate methods)

### Similarity Metric
- **Choice**: Cosine similarity (implemented via normalized inner product)
- **Advantages**: Standard for text embeddings, handles different document lengths well

### Generation Model
- **Choice**: Gemini Pro via Google Generative AI
- **Advantages**: Free tier available, good reasoning capabilities, context-aware responses
- **Tradeoff**: API rate limits and dependency on external service

## Manual vs Framework Approach

### Advantages of Manual Implementation:
1. **Full Control**: Complete understanding and control over each component
2. **Customization**: Easy to modify specific parts without framework constraints
3. **Learning**: Deep understanding of RAG internals
4. **Lightweight**: No unnecessary abstractions or dependencies
5. **Transparency**: Clear visibility into data flow and transformations

### Disadvantages:
1. **Development Time**: More time to implement from scratch
2. **Error Prone**: Manual implementation may have bugs
3. **Limited Features**: Missing advanced features like reranking, hybrid search
4. **Maintenance**: Need to handle updates and improvements manually
5. **Scalability**: Would need significant work to scale to production

### When to Use Each Approach:
- **Manual**: Learning, small datasets, specific requirements, full control needed
- **Framework**: Production systems, large datasets, rapid development, complex features

## Performance Characteristics

- **Indexing Time**: ~30-60 seconds for 100 emails
- **Query Latency**: ~1-3 seconds (including API call)
- **Memory Usage**: ~50-100MB (embeddings + index)
- **Accuracy**: High for simple queries, depends on question complexity

## Limitations

1. **Single Modal**: Text-only, no support for images or attachments
2. **No Reranking**: Uses simple similarity search only
3. **Limited Context**: Fixed context window for generation
4. **No Persistence**: Index rebuilt each session (can be added)
5. **Single User**: No concurrent query handling

## Future Improvements

1. **Hybrid Search**: Combine semantic and keyword search
2. **Reranking**: Add cross-encoder for better relevance
3. **Persistence**: Save/load index to disk
4. **Evaluation**: Add automatic relevance metrics
5. **Streaming**: Process large datasets incrementally
6. **Multi-modal**: Handle different document types

## File Structure

```
├── rag_pipeline.py          # Main pipeline orchestrator
├── document_processor.py     # Email loading and chunking
├── embedding_engine.py       # Embedding generation
├── retrieval_engine.py       # FAISS-based retrieval
├── generator.py             # Answer generation with Gemini
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables (API key)
├── emails/                 # Dataset directory
│   ├── email_001.txt
│   ├── email_002.txt
│   └── ...
└── README.md               # This file
```
