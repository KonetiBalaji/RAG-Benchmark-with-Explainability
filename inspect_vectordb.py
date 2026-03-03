"""Inspect ChromaDB vector database contents.

This script helps you view what's stored in the vector database.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.vector_store import VectorStore
from src.utils.config_loader import get_config
import pandas as pd
from loguru import logger


def inspect_vector_db():
    """Inspect the vector database and show stored data."""
    
    print("\n" + "="*80)
    print("VECTOR DATABASE INSPECTOR")
    print("="*80)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Get basic stats
    count = vector_store.get_count()
    print(f"\nTotal Documents in Vector DB: {count:,}")
    
    if count == 0:
        print("\nVector database is empty!")
        print("Run this to build it: python main.py build-index")
        return
    
    # Get collection info
    collection = vector_store.collection
    print(f"Collection Name: {collection.name}")
    print(f"Distance Metric: cosine similarity")
    
    # Sample some documents
    print("\n" + "-"*80)
    print("SAMPLE DOCUMENTS (First 10)")
    print("-"*80)
    
    # Get first 10 documents
    results = collection.get(limit=10, include=['documents', 'metadatas', 'embeddings'])
    
    for i, (doc_id, doc_text, metadata, embedding) in enumerate(zip(
        results['ids'],
        results['documents'],
        results['metadatas'],
        results['embeddings']
    ), 1):
        print(f"\n[{i}] ID: {doc_id}")
        print(f"    Text: {doc_text[:150]}...")
        print(f"    Metadata: {metadata}")
        print(f"    Embedding dim: {len(embedding) if embedding else 'N/A'}")
        print(f"    First 5 values: {embedding[:5] if embedding else 'N/A'}")
    
    # Get all data for analysis
    print("\n" + "-"*80)
    print("STATISTICS")
    print("-"*80)
    
    all_results = collection.get(include=['documents', 'metadatas'])
    
    # Document length stats
    doc_lengths = [len(doc) for doc in all_results['documents']]
    print(f"\nDocument Text Lengths:")
    print(f"  Min: {min(doc_lengths)} chars")
    print(f"  Max: {max(doc_lengths)} chars")
    print(f"  Avg: {sum(doc_lengths)/len(doc_lengths):.1f} chars")
    
    # Metadata analysis
    if all_results['metadatas']:
        print(f"\nMetadata Fields:")
        all_keys = set()
        for meta in all_results['metadatas']:
            if meta:
                all_keys.update(meta.keys())
        for key in sorted(all_keys):
            print(f"  - {key}")
    
    # Export option
    print("\n" + "-"*80)
    print("EXPORT OPTIONS")
    print("-"*80)
    
    choice = input("\nExport data to CSV? (y/n): ").lower()
    if choice == 'y':
        export_to_csv(all_results)


def export_to_csv(results):
    """Export vector DB contents to CSV."""
    
    # Create DataFrame
    data = []
    for doc_id, doc_text, metadata in zip(
        results['ids'],
        results['documents'],
        results['metadatas']
    ):
        row = {
            'id': doc_id,
            'text': doc_text,
            'text_length': len(doc_text),
        }
        # Add metadata fields
        if metadata:
            row.update(metadata)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = "vectordb_export.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nExported {len(df)} documents to: {output_file}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())


def search_vector_db(query_text):
    """Search the vector database with a query."""
    
    print("\n" + "="*80)
    print("VECTOR DATABASE SEARCH")
    print("="*80)
    
    from src.data.embedding_generator import EmbeddingGenerator
    
    vector_store = VectorStore()
    embedding_gen = EmbeddingGenerator()
    
    # Generate query embedding
    print(f"\nQuery: {query_text}")
    query_embedding = embedding_gen.generate_embedding(query_text)
    
    # Search
    ids, scores, documents, metadatas = vector_store.search(
        query_embedding=query_embedding,
        top_k=5
    )
    
    print("\nTop 5 Results:")
    print("-"*80)
    
    for i, (doc_id, score, text, metadata) in enumerate(zip(ids, scores, documents, metadatas), 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"    ID: {doc_id}")
        print(f"    Text: {text[:200]}...")
        print(f"    Metadata: {metadata}")


def show_embeddings_sample():
    """Show sample embeddings to understand the vector representation."""
    
    print("\n" + "="*80)
    print("EMBEDDING VISUALIZATION")
    print("="*80)
    
    vector_store = VectorStore()
    
    # Get one document with embedding
    results = vector_store.collection.get(limit=1, include=['documents', 'embeddings'])
    
    if results['embeddings']:
        embedding = results['embeddings'][0]
        text = results['documents'][0]
        
        print(f"\nDocument: {text[:100]}...")
        print(f"\nEmbedding (1536 dimensions):")
        print(f"  First 20 values: {embedding[:20]}")
        print(f"  Shape: {len(embedding)}")
        print(f"  Min value: {min(embedding):.6f}")
        print(f"  Max value: {max(embedding):.6f}")
        print(f"  Mean: {sum(embedding)/len(embedding):.6f}")
        
        # Show distribution
        import statistics
        print(f"  Std dev: {statistics.stdev(embedding):.6f}")


def interactive_inspector():
    """Interactive inspector with menu."""
    
    while True:
        print("\n" + "="*80)
        print("VECTOR DATABASE INSPECTOR - MENU")
        print("="*80)
        print("\n1. Show database overview")
        print("2. View sample documents")
        print("3. Search database")
        print("4. View embeddings")
        print("5. Export to CSV")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            inspect_vector_db()
        elif choice == '2':
            show_sample_docs()
        elif choice == '3':
            query = input("Enter search query: ")
            search_vector_db(query)
        elif choice == '4':
            show_embeddings_sample()
        elif choice == '5':
            vector_store = VectorStore()
            results = vector_store.collection.get(include=['documents', 'metadatas'])
            export_to_csv(results)
        elif choice == '6':
            print("\nGoodbye!")
            break
        else:
            print("Invalid option. Try again.")


def show_sample_docs():
    """Show sample documents in detail."""
    
    print("\n" + "="*80)
    print("SAMPLE DOCUMENTS")
    print("="*80)
    
    num = int(input("\nHow many documents to show? (1-50): "))
    num = min(max(num, 1), 50)
    
    vector_store = VectorStore()
    results = vector_store.collection.get(limit=num, include=['documents', 'metadatas'])
    
    for i, (doc_id, text, metadata) in enumerate(zip(
        results['ids'],
        results['documents'],
        results['metadatas']
    ), 1):
        print(f"\n{'='*80}")
        print(f"Document {i}/{num}")
        print(f"{'='*80}")
        print(f"ID: {doc_id}")
        print(f"Metadata: {metadata}")
        print(f"\nFull Text:\n{text}")
        
        if i < num:
            cont = input("\nPress Enter for next, 'q' to quit: ")
            if cont.lower() == 'q':
                break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            search_vector_db(query)
        elif sys.argv[1] == "export":
            vector_store = VectorStore()
            results = vector_store.collection.get(include=['documents', 'metadatas'])
            export_to_csv(results)
        elif sys.argv[1] == "stats":
            inspect_vector_db()
        else:
            print("Usage:")
            print("  python inspect_vectordb.py              # Interactive mode")
            print("  python inspect_vectordb.py stats        # Show statistics")
            print("  python inspect_vectordb.py search <query>  # Search database")
            print("  python inspect_vectordb.py export       # Export to CSV")
    else:
        # Interactive mode
        interactive_inspector()
