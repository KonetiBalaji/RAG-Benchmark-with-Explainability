"""Quick test script to verify RAG system with sample data.

Run this before full benchmark to test all 4 configurations quickly.
"""

import sys
from pathlib import Path
from loguru import logger

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.data.sample_data import SampleDataGenerator
from src.data.text_chunker import TextChunker
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStoreManager
from src.models.baseline_rag import BaselineRAG
from src.guardrails.guardrail_checker import GuardrailChecker


def main():
    """Run quick test with sample data."""
    print("=" * 70)
    print("RAG System Quick Test - Using Sample Data")
    print("=" * 70)
    print()
    
    # Setup
    setup_logger()
    config = get_config()
    
    # Step 1: Generate sample data
    print("Step 1: Generating sample data...")
    generator = SampleDataGenerator()
    generator.generate()
    sample_data = generator.load_sample_data()
    print(f"[OK] Loaded {len(sample_data['queries'])} queries and {len(sample_data['passages'])} passages\n")
    
    # Step 2: Chunk documents
    print("Step 2: Chunking documents...")
    chunker = TextChunker(config)
    chunks = chunker.chunk_documents(sample_data['passages'])
    print(f"[OK] Created {len(chunks)} chunks\n")
    
    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    print("[WARNING] This will use OpenAI API and incur small cost (~$0.001)")
    
    try:
        embedding_gen = EmbeddingGenerator(config)
        embeddings = embedding_gen.generate_batch(chunks)
        print(f"[OK] Generated {len(embeddings)} embeddings\n")
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        print("\nMake sure OPENAI_API_KEY is set in .env file")
        return 1
    
    # Step 4: Create vector store
    print("Step 4: Creating vector store...")
    try:
        # Use a test collection
        config._config['vector_db']['collection_name'] = 'test_collection'
        vector_store = VectorStoreManager(config)
        vector_store.add_documents(chunks, embeddings)
        print(f"[OK] Vector store created with {len(chunks)} documents\n")
    except Exception as e:
        print(f"✗ Vector store creation failed: {e}")
        return 1
    
    # Step 5: Test baseline RAG
    print("Step 5: Testing Baseline RAG...")
    try:
        rag = BaselineRAG(config, vector_store)
        test_query = sample_data['queries'][0]['query']
        print(f"Query: {test_query}")
        
        result = rag.query(test_query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nRetrieved {len(result['retrieved_docs'])} documents:")
        for i, doc in enumerate(result['retrieved_docs'][:2], 1):
            print(f"  {i}. Score: {doc['score']:.3f}")
            print(f"     {doc['text'][:100]}...")
        
        print()
    except Exception as e:
        print(f"✗ RAG query failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 6: Test guardrails
    print("Step 6: Testing hallucination guardrails...")
    try:
        guardrail = GuardrailChecker(config)
        
        # Test with good retrieval scores
        check_result = guardrail.check(
            query=test_query,
            answer=result['answer'],
            retrieved_docs=result['retrieved_docs']
        )
        
        print(f"Confidence: {check_result['confidence']:.2f}")
        print(f"Should answer: {check_result['should_answer']}")
        print(f"Warning: {check_result['warning'] or 'None'}")
        print()
    except Exception as e:
        print(f"[WARNING] Guardrail check failed: {e}")
        print("Continuing...\n")
    
    # Summary
    print("=" * 70)
    print("[SUCCESS] Quick Test Complete!")
    print("=" * 70)
    print("\nThe system is working correctly. Next steps:")
    print("  1. python main.py prepare-data --sample    # Prepare sample data")
    print("  2. python main.py benchmark                # Run full benchmark")
    print("  3. streamlit run src/ui/app.py             # Launch UI")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
