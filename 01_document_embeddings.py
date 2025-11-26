"""Document Embeddings - Beginner's Guide

This script demonstrates how to:
1. Load a pre-trained sentence transformer model
2. Generate embeddings for text documents
3. Calculate similarity between documents
4. Find the most similar documents to a query

Author: Vector Embedding Learning Project
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model_name='all-MiniLM-L6-v2'):
    """Load a pre-trained sentence transformer model.
    
    Args:
        model_name: Name of the model to load
            - 'all-MiniLM-L6-v2': Fast, 384 dimensions
            - 'all-mpnet-base-v2': Higher quality, 768 dimensions
    
    Returns:
        Loaded SentenceTransformer model
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(model, documents):
    """Generate embeddings for a list of documents.
    
    Args:
        model: Loaded SentenceTransformer model
        documents: List of text documents
    
    Returns:
        numpy array of shape (n_documents, embedding_dim)
    """
    print(f"\nGenerating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def calculate_similarity_matrix(embeddings):
    """Calculate pairwise cosine similarity between all embeddings.
    
    Args:
        embeddings: numpy array of embeddings
    
    Returns:
        Similarity matrix of shape (n_documents, n_documents)
    """
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def find_most_similar(query, documents, model, embeddings, top_k=3):
    """Find the most similar documents to a query.
    
    Args:
        query: Query text
        documents: List of documents
        model: SentenceTransformer model
        embeddings: Pre-computed document embeddings
        top_k: Number of results to return
    
    Returns:
        List of (document, similarity_score) tuples
    """
    # Generate query embedding
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [(documents[i], similarities[i]) for i in top_indices]
    return results


def main():
    # Sample documents for demonstration
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with many layers to model complex patterns in data.",
        "Python is a popular programming language for data science and machine learning.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
        "Data preprocessing is essential for cleaning and preparing data for analysis.",
        "Feature engineering involves creating new features from existing data to improve model performance.",
    ]
    
    # Load model
    model = load_model()
    
    # Generate embeddings
    embeddings = generate_embeddings(model, documents)
    
    # Calculate similarity matrix
    print("\n--- Document Similarity Matrix ---")
    similarity_matrix = calculate_similarity_matrix(embeddings)
    
    # Print similarity between first few documents
    print("\nSimilarity between documents (sample):")
    for i in range(min(3, len(documents))):
        for j in range(i+1, min(4, len(documents))):
            print(f"Doc {i} <-> Doc {j}: {similarity_matrix[i][j]:.4f}")
    
    # Semantic search example
    print("\n--- Semantic Search Demo ---")
    query = "How do neural networks learn?"
    print(f"Query: '{query}'")
    print("\nMost similar documents:")
    
    results = find_most_similar(query, documents, model, embeddings, top_k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc[:80]}..." if len(doc) > 80 else f"  [{score:.4f}] {doc}")
    
    # Another query
    print("\n" + "="*50)
    query2 = "preparing data for machine learning"
    print(f"Query: '{query2}'")
    print("\nMost similar documents:")
    
    results2 = find_most_similar(query2, documents, model, embeddings, top_k=3)
    for doc, score in results2:
        print(f"  [{score:.4f}] {doc[:80]}..." if len(doc) > 80 else f"  [{score:.4f}] {doc}")


if __name__ == "__main__":
    main()
