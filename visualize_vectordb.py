"""Visualize vector database using dimensionality reduction.

Creates 2D/3D visualizations of the high-dimensional embeddings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.vector_store import VectorStore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def visualize_embeddings_2d(method='pca', max_points=1000):
    """Visualize embeddings in 2D using PCA or t-SNE.
    
    Args:
        method: 'pca' or 'tsne'
        max_points: Maximum number of points to plot
    """
    
    print(f"\nVisualizing embeddings using {method.upper()}...")
    
    # Load vector store
    vector_store = VectorStore()
    
    # Get embeddings
    results = vector_store.collection.get(
        limit=max_points,
        include=['embeddings', 'documents', 'metadatas']
    )
    
    if not results['embeddings']:
        print("No embeddings found! Run: python main.py build-index")
        return
    
    embeddings = np.array(results['embeddings'])
    documents = results['documents']
    
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Reduce dimensions
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
        title = f"Vector Database Embeddings (PCA)\nExplained variance: {sum(reducer.explained_variance_ratio_):.1%}"
    else:  # tsne
        print("Running t-SNE (this may take a minute)...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords_2d = reducer.fit_transform(embeddings)
        title = "Vector Database Embeddings (t-SNE)"
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot points
    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        alpha=0.6,
        s=50,
        c=range(len(coords_2d)),
        cmap='viridis'
    )
    
    # Add labels for a few points
    for i in range(min(10, len(documents))):
        label = documents[i][:30] + "..." if len(documents[i]) > 30 else documents[i]
        plt.annotate(
            label,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.colorbar(scatter, label='Document Index')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    
    # Save figure
    output_file = f'vectordb_visualization_{method}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.show()


def visualize_query_results(query_text, top_k=10):
    """Visualize query and its results in embedding space."""
    
    print(f"\nVisualizing query results for: '{query_text}'")
    
    from src.data.embedding_generator import EmbeddingGenerator
    
    # Get vector store and generate query embedding
    vector_store = VectorStore()
    embedding_gen = EmbeddingGenerator()
    
    query_embedding = embedding_gen.generate_embedding(query_text)
    
    # Search
    ids, scores, documents, _ = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k
    )
    
    # Get all embeddings for context
    all_results = vector_store.collection.get(
        limit=500,
        include=['embeddings', 'documents']
    )
    
    all_embeddings = np.array(all_results['embeddings'])
    all_docs = all_results['documents']
    
    # Add query embedding
    all_embeddings = np.vstack([query_embedding, all_embeddings])
    all_docs = [f"QUERY: {query_text}"] + all_docs
    
    # Reduce to 2D
    print("Reducing dimensions with PCA...")
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(all_embeddings)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Plot all documents (gray)
    plt.scatter(
        coords_2d[1:, 0],
        coords_2d[1:, 1],
        alpha=0.3,
        s=30,
        c='gray',
        label='Other documents'
    )
    
    # Plot query (red star)
    plt.scatter(
        coords_2d[0, 0],
        coords_2d[0, 1],
        alpha=1.0,
        s=300,
        c='red',
        marker='*',
        label='Query',
        edgecolors='black',
        linewidths=2
    )
    
    # Find and highlight retrieved documents
    retrieved_indices = []
    for retrieved_id in ids[:top_k]:
        # Find index in all_docs
        for idx, doc in enumerate(all_docs[1:], 1):
            if doc == documents[ids.index(retrieved_id)]:
                retrieved_indices.append(idx)
                break
    
    if retrieved_indices:
        plt.scatter(
            coords_2d[retrieved_indices, 0],
            coords_2d[retrieved_indices, 1],
            alpha=0.8,
            s=100,
            c='green',
            marker='o',
            label=f'Top {top_k} results',
            edgecolors='darkgreen',
            linewidths=1.5
        )
    
    # Annotate query
    plt.annotate(
        query_text,
        (coords_2d[0, 0], coords_2d[0, 1]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
    )
    
    plt.legend()
    plt.title(f'Query Results Visualization\nQuery: "{query_text}"')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.tight_layout()
    
    # Save
    output_file = 'query_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()


def show_embedding_clusters(n_clusters=5):
    """Show clusters in the embedding space."""
    
    print(f"\nClustering embeddings into {n_clusters} groups...")
    
    from sklearn.cluster import KMeans
    
    vector_store = VectorStore()
    results = vector_store.collection.get(
        limit=1000,
        include=['embeddings', 'documents']
    )
    
    embeddings = np.array(results['embeddings'])
    documents = results['documents']
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings)
    centroids_2d = pca.transform(kmeans.cluster_centers_)
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=50
    )
    
    # Plot centroids
    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c='red',
        marker='X',
        s=300,
        edgecolors='black',
        linewidths=2,
        label='Cluster centers'
    )
    
    # Show sample docs from each cluster
    for cluster_id in range(n_clusters):
        cluster_docs = [doc for doc, label in zip(documents, labels) if label == cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_docs)} documents")
        print(f"  Sample: {cluster_docs[0][:80]}...")
    
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.title(f'Embedding Clusters (k={n_clusters})')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.tight_layout()
    
    output_file = 'embedding_clusters.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.show()


if __name__ == "__main__":
    print("="*80)
    print("VECTOR DATABASE VISUALIZATION")
    print("="*80)
    print("\nOptions:")
    print("1. 2D PCA visualization")
    print("2. 2D t-SNE visualization")
    print("3. Query results visualization")
    print("4. Cluster analysis")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        visualize_embeddings_2d(method='pca')
    elif choice == '2':
        visualize_embeddings_2d(method='tsne')
    elif choice == '3':
        query = input("Enter query: ")
        visualize_query_results(query)
    elif choice == '4':
        n = int(input("Number of clusters (default 5): ") or "5")
        show_embedding_clusters(n)
    else:
        print("Invalid option")
