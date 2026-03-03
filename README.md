# RAG System Benchmark with Explainability and Guardrails

A production-ready Retrieval-Augmented Generation (RAG) system with comprehensive benchmarking, explainability features, and hallucination guardrails.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Custom Data Upload](#custom-data-upload)
- [RAG Configurations](#rag-configurations)
- [Evaluation Metrics](#evaluation-metrics)
- [Guardrails](#guardrails)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This system implements and benchmarks multiple RAG strategies with built-in explainability and hallucination prevention. It provides a transparent view into how answers are generated, what sources were used, and the confidence level of each response.

### Key Capabilities

- **4 RAG Strategies**: Baseline, Hybrid (BM25+Semantic), Reranker, Query Decomposition
- **Hallucination Prevention**: Multi-layer guardrails with confidence scoring
- **Explainability**: Full transparency into retrieved context and reasoning
- **Custom Data Support**: Upload PDF, DOCX, TXT, JSON, JSONL, CSV files
- **Comprehensive Metrics**: Retrieval (Precision@k, MRR), Generation (ROUGE, RAGAS), Operational (latency, cost)
- **Interactive UI**: Streamlit-based interface with side-by-side model comparison
- **Cost Tracking**: Real-time monitoring of API costs

## Architecture

```
src/
├── models/          # RAG implementations
│   ├── base_rag.py           # Abstract base class
│   ├── baseline_rag.py       # Simple semantic search
│   ├── hybrid_rag.py         # BM25 + Semantic
│   ├── reranker_rag.py       # With Cohere reranking
│   └── query_decomposition_rag.py
├── data/            # Data loading and processing
│   ├── data_loader.py        # MS MARCO & custom file support
│   ├── text_chunker.py       # Recursive text splitting
│   ├── embedding_generator.py
│   └── vector_store.py       # ChromaDB integration
├── evaluation/      # Metrics and benchmarking
│   ├── metrics.py            # Precision, MRR, ROUGE
│   ├── ragas_metrics.py      # RAGAS framework
│   └── benchmark.py          # Full evaluation suite
├── guardrails/      # Hallucination prevention
│   └── guardrail_checker.py  # Multi-layer checks
├── ui/              # User interface
│   └── app.py                # Streamlit application
├── api/             # REST API
│   └── main.py               # FastAPI endpoints
└── utils/           # Utilities
    ├── config_loader.py
    ├── cost_tracker.py
    └── logger.py
```

### Design Principles

1. **Modular Architecture**: Clean separation of concerns following SOLID principles
2. **Abstract Base Classes**: Easy to extend with new RAG strategies
3. **Configuration-Driven**: All parameters externalized in config.yaml
4. **Production-Ready**: Logging, monitoring, error handling, cost tracking
5. **Explainable AI**: Full transparency into retrieval and generation process

## Features

### RAG Strategies Implemented

1. **Baseline RAG**
   - Simple semantic search using OpenAI embeddings
   - Cosine similarity matching
   - Best for: Simple queries, well-defined topics

2. **Hybrid RAG**
   - Combines BM25 (keyword) + Semantic (dense) search
   - Reciprocal Rank Fusion for result merging
   - Best for: Keyword-specific queries, technical terms

3. **Reranker RAG**
   - Initial semantic retrieval followed by Cohere reranking
   - Cross-encoder reranking for better relevance
   - Best for: Complex queries requiring nuanced understanding

4. **Query Decomposition RAG**
   - Breaks complex queries into sub-questions
   - Retrieves context for each sub-question
   - Synthesizes final answer
   - Best for: Multi-part questions, reasoning tasks

### Hallucination Guardrails

Two-layer protection system:

1. **Retrieval Score Threshold** (Primary)
   - Minimum similarity score: 0.6
   - Prevents answering when retrieved context is irrelevant
   - Fast, lightweight check

2. **NLI Entailment Check** (Optional)
   - Uses facebook/bart-large-mnli model
   - Verifies answer is entailed by retrieved context
   - Currently disabled (too conservative for demo)

Confidence Levels:
- **High (≥0.8)**: Strong evidence in retrieved context
- **Medium (0.6-0.8)**: Moderate support
- **Low (<0.6)**: Guardrail triggered - refuses to answer

### Custom Data Upload

Upload and analyze your own documents in multiple formats:

**Supported Formats:**
- **PDF**: Research papers, reports, documentation
- **DOCX**: Business documents, meeting notes
- **TXT**: Plain text files (one doc per line/paragraph)
- **JSON**: Structured data `{"documents": [...], "queries": [...]}`
- **JSONL**: Line-delimited JSON objects
- **CSV**: Tabular data with 'text' column

**How It Works:**
1. Upload file via UI
2. System extracts and chunks text
3. Generates embeddings using OpenAI text-embedding-3-small
4. Builds ChromaDB vector index
5. All RAG models ready to query your data

**Example:**
```python
from src.data.data_loader import DatasetLoader

loader = DatasetLoader()
queries_df, passages_df = loader.load_from_file("research_paper.pdf")
```

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Cohere API key (for reranker)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/rag-benchmark.git
cd rag-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here
```

### Run the System

```bash
# Option 1: Interactive UI (Recommended)
python main.py ui

# Option 2: Run benchmark evaluation
python main.py benchmark

# Option 3: Quick test
python main.py quick-test
```

## Usage

### Using the UI

1. Start the Streamlit interface:
```bash
streamlit run src/ui/app.py
```

2. Navigate to `http://localhost:8501`

3. Select data source:
   - **MS MARCO Dataset**: Pre-loaded 10,000 passages
   - **Upload Custom File**: Your own documents

4. Choose RAG configuration:
   - Single Model: Test one strategy
   - Compare Models: Side-by-side comparison

5. Enter query and get explainable answers with:
   - Retrieved evidence chunks
   - Similarity scores
   - Confidence level
   - Guardrail status
   - Performance metrics

### Demo Queries (MS MARCO Dataset)

**Query 1: Manhattan Project (High Confidence)**
```
what was the Manhattan Project?
```
Expected: Confidence 74.5%, detailed historical answer, guardrail PASSED

**Query 2: Corporation (Medium Confidence)**
```
what is a corporation?
```
Expected: Confidence 39%, partial answer, guardrail may TRIGGER

**Query 3: Python ML Code (Guardrail Demo)**
```
how to train a neural network in python?
```
Expected: Confidence 23.3%, guardrail TRIGGERED, refuses to answer

### Programmatic Usage

```python
from src.models.baseline_rag import BaselineRAG
from src.data.vector_store import VectorStore
from src.guardrails.guardrail_checker import GuardrailChecker

# Initialize components
vector_store = VectorStore()
rag = BaselineRAG(vector_store)
guardrails = GuardrailChecker()

# Get answer
response = rag.answer("What is machine learning?", top_k=3)

# Check guardrails
triggered, reason, details = guardrails.check_guardrails(
    query=response.query,
    chunks=response.retrieved_chunks,
    answer=response.answer
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score:.2%}")
print(f"Guardrail: {'TRIGGERED' if triggered else 'PASSED'}")
```

## Custom Data Upload

### File Format Examples

**1. JSON Format**
```json
{
  "documents": [
    "Python is a high-level programming language.",
    "Machine learning is a subset of AI."
  ],
  "queries": [
    "What is Python?",
    "Explain machine learning"
  ]
}
```

**2. CSV Format**
```csv
text,metadata
"Document 1 content","category1"
"Document 2 content","category2"
```

**3. TXT Format**
```
Document 1 content goes here.

Document 2 content separated by double newline.
```

### Upload Workflow

1. **Select "Upload Custom File"** in sidebar
2. **Choose file** (PDF, DOCX, TXT, JSON, JSONL, CSV)
3. **Wait for processing** (chunking + embedding generation)
4. **Query your data** using any RAG model

### Tips for Best Results

- Keep documents focused on single topics
- Use clear, well-formatted text
- For PDFs: Use text-based (not scanned images)
- For large datasets: Consider JSONL format
- Start small (few documents) then scale up
- Test with example queries included in JSON

## RAG Configurations

### Baseline Configuration

```yaml
rag_configs:
  baseline:
    name: "Baseline Semantic Search"
    retrieval_type: "semantic"
    top_k: 3
```

**Implementation:** `src/models/baseline_rag.py`
- Uses OpenAI text-embedding-3-small (1536 dimensions)
- Cosine similarity in ChromaDB
- Simple, fast, effective for most queries

### Hybrid Configuration

```yaml
rag_configs:
  hybrid:
    name: "Hybrid Search (BM25 + Semantic)"
    retrieval_type: "hybrid"
    top_k: 10
    bm25_weight: 0.5
    semantic_weight: 0.5
    final_top_k: 3
```

**Implementation:** `src/models/hybrid_rag.py`
- BM25 for keyword matching
- Semantic search for meaning
- Reciprocal Rank Fusion (RRF) for merging

### Reranker Configuration

```yaml
rag_configs:
  reranker:
    name: "With Cohere Reranker"
    retrieval_type: "semantic"
    top_k: 10
    reranker_model: "rerank-english-v3.0"
    reranker_top_k: 3
```

**Implementation:** `src/models/reranker_rag.py`
- Initial retrieval: 10 candidates
- Cohere cross-encoder reranking
- Final selection: Top 3

### Query Decomposition Configuration

```yaml
rag_configs:
  query_decomposition:
    name: "Query Decomposition"
    retrieval_type: "semantic"
    decomposition_model: "gpt-3.5-turbo"
    max_subqueries: 3
    top_k_per_subquery: 2
```

**Implementation:** `src/models/query_decomposition_rag.py`
- Breaks complex queries into sub-questions
- Retrieves context for each
- Synthesizes comprehensive answer

## Evaluation Metrics

### Retrieval Metrics

- **Precision@k**: Proportion of relevant docs in top-k results
- **Recall@k**: Proportion of relevant docs retrieved
- **MRR (Mean Reciprocal Rank)**: Position of first relevant doc
- **NDCG**: Normalized discounted cumulative gain

### Generation Metrics

- **ROUGE-L**: Longest common subsequence overlap
- **Faithfulness**: Answer grounded in retrieved context
- **Answer Relevancy**: Answer addresses the query
- **Context Precision**: Relevant context in top positions

### Operational Metrics

- **Latency**: End-to-end response time (ms)
- **Cost**: API usage in USD
- **Token Count**: Input + output tokens
- **Guardrail Trigger Rate**: % of queries triggering guardrails

### Running Benchmarks

```bash
# Full benchmark across all models
python main.py benchmark

# Quick test (subset of queries)
python main.py quick-test

# Generate evaluation report
python -c "from src.evaluation.benchmark import RAGBenchmark; b = RAGBenchmark(); b.run()"
```

## Guardrails

### Configuration

```yaml
guardrails:
  retrieval_threshold: 0.6    # Min similarity score
  nli_enabled: false          # Disable NLI (too strict)
  nli_model: "facebook/bart-large-mnli"
  nli_threshold: 0.5          # Min entailment probability
  confidence_levels:
    high: 0.8
    medium: 0.6
    low: 0.4
```

### How Guardrails Work

1. **Retrieval Check**
   ```python
   max_score = max(chunk.score for chunk in retrieved_chunks)
   if max_score < retrieval_threshold:
       return "I don't have enough confident information..."
   ```

2. **NLI Check** (Optional)
   ```python
   entailment_score = nli_model(context, answer)
   if entailment_score < nli_threshold:
       return "Answer may not be fully supported by context..."
   ```

3. **Confidence Level**
   ```python
   if max_score >= 0.8: confidence = "HIGH"
   elif max_score >= 0.6: confidence = "MEDIUM"
   else: confidence = "LOW" (guardrail triggered)
   ```

### Customizing Guardrails

Edit `configs/config.yaml`:
- Lower `retrieval_threshold` for more permissive answers
- Enable `nli_enabled: true` for stricter fact-checking
- Adjust `confidence_levels` thresholds

## API Documentation

### REST API Endpoints

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload
```

**POST /query**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "model": "baseline",
    "top_k": 3
  }'
```

Response:
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is...",
  "confidence_score": 0.85,
  "guardrail_triggered": false,
  "retrieved_chunks": [...],
  "metadata": {
    "latency_ms": 1234,
    "cost_usd": 0.0012
  }
}
```

**GET /health**
```bash
curl "http://localhost:8000/health"
```

## Configuration

### Main Configuration File: `configs/config.yaml`

**Dataset Settings**
```yaml
dataset:
  name: "msmarco"
  num_queries: 500
  num_passages: 10000
```

**Embedding Settings**
```yaml
embeddings:
  model: "text-embedding-3-small"
  dimensions: 1536
  batch_size: 100
```

**LLM Settings**
```yaml
llm:
  model: "gpt-3.5-turbo"
  temperature: 0.0  # Deterministic
  max_tokens: 512
  seed: 42
```

**Vector Database**
```yaml
vector_db:
  type: "chroma"
  persist_directory: "./data/vector_db"
  distance_metric: "cosine"
```

### Environment Variables

Create `.env` file:
```bash
# Required
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...

# Optional
LOG_LEVEL=INFO
CACHE_DIR=./data/cache
```

## Testing

### Run All Tests

```bash
# Unit tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_rag_models.py
```

### Test Coverage

Current coverage: ~60%
- `tests/test_config.py`: Configuration loading
- `tests/test_metrics.py`: Evaluation metrics
- `tests/test_rag_models.py`: RAG implementations

### Quick Integration Test

```bash
python main.py quick-test
```

Runs end-to-end test:
1. Load sample data
2. Build vector index
3. Query with all 4 RAG models
4. Validate responses
5. Generate metrics report

## Deployment

### Local Deployment

```bash
# Activate environment
source venv/bin/activate

# Start UI
streamlit run src/ui/app.py --server.port 8501
```

### Docker Deployment

```bash
# Build image
docker build -t rag-benchmark .

# Run container
docker run -p 8501:8501 --env-file .env rag-benchmark
```

### Production Checklist

- [ ] Set `temperature=0` for deterministic outputs
- [ ] Enable guardrails (`retrieval_threshold: 0.6`)
- [ ] Configure rate limiting
- [ ] Set up monitoring (cost tracking, latency)
- [ ] Use production-grade vector DB (e.g., Pinecone, Weaviate)
- [ ] Implement caching for frequent queries
- [ ] Add authentication to API endpoints
- [ ] Set up CI/CD pipeline
- [ ] Configure logging to external service
- [ ] Implement health checks

## Industry Best Practices Applied

### Code Quality
- Abstract base classes for extensibility
- Type hints throughout
- Docstrings following Google style
- Modular, single-responsibility design

### MLOps
- Configuration externalization
- Comprehensive metrics tracking
- Cost monitoring
- Reproducible experiments (seed=42)

### Production Readiness
- Structured logging (loguru)
- Error handling
- Guardrails for safety
- Explainability for trust

### Data Handling
- Supports multiple file formats
- Efficient chunking strategy
- Batch processing for embeddings
- Persistent vector storage

## Troubleshooting

### Common Issues

**Issue: "OpenAI API key not found"**
Solution: Create `.env` file with `OPENAI_API_KEY=your_key`

**Issue: "Vector store is empty"**
Solution: Run `python main.py build-index` first

**Issue: "PDF support not available"**
Solution: `pip install pypdf python-docx`

**Issue: "Guardrails too strict"**
Solution: Lower `retrieval_threshold` in config.yaml

**Issue: "Out of memory"**
Solution: Reduce `num_passages` or `batch_size` in config

## Contributing

Contributions welcome! Please follow:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- MS MARCO dataset (Microsoft)
- OpenAI embeddings and GPT models
- Cohere reranking API
- RAGAS evaluation framework
- ChromaDB vector database
- Streamlit UI framework

---

**Built with industry best practices for production RAG systems**
