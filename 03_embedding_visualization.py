"""Embedding Visualization - Beginner's Guide

This script demonstrates how to:
1. Reduce high-dimensional embeddings to 2D/3D using t-SNE and UMAP
2. Create interactive 2D scatter plots with matplotlib
3. Create interactive 3D plots with Plotly
4. Color-code clusters for better understanding

Author: Vector Embedding Learning Project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Try to import plotly for interactive 3D plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed. 3D interactive plots disabled.")
    print("Install with: pip install plotly")


def create_sample_data():
    """Create sample documents grouped by category."""
    data = {
        'AI/ML': [
            "Machine learning algorithms learn patterns from data",
            "Neural networks are inspired by biological brains",
            "Deep learning requires large amounts of training data",
            "Artificial intelligence is transforming industries",
        ],
        'Programming': [
            "Python is excellent for data science applications",
            "JavaScript powers interactive web applications",
            "Java is widely used in enterprise software",
            "Rust provides memory safety without garbage collection",
        ],
        'Database': [
            "SQL databases use structured query language",
            "MongoDB is a popular NoSQL document database",
            "Redis provides in-memory data structure storage",
            "PostgreSQL supports advanced SQL features",
        ],
        'Cloud': [
            "AWS provides comprehensive cloud services",
            "Kubernetes orchestrates container deployments",
            "Docker containers package applications consistently",
            "Serverless computing scales automatically",
        ],
    }
    
    documents = []
    labels = []
    for category, docs in data.items():
        documents.extend(docs)
        labels.extend([category] * len(docs))
    
    return documents, labels


def generate_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for documents."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings


def reduce_dimensions_tsne(embeddings, n_components=2, perplexity=5):
    """Reduce dimensions using t-SNE."""
    print(f"\nReducing to {n_components}D using t-SNE...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000
    )
    reduced = tsne.fit_transform(embeddings)
    print(f"Reduced shape: {reduced.shape}")
    return reduced


def reduce_dimensions_pca(embeddings, n_components=2):
    """Reduce dimensions using PCA."""
    print(f"\nReducing to {n_components}D using PCA...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    variance_ratio = pca.explained_variance_ratio_
    print(f"Reduced shape: {reduced.shape}")
    print(f"Explained variance: {variance_ratio}")
    return reduced


def plot_2d_matplotlib(reduced_embeddings, labels, documents, title="Embeddings 2D"):
    """Create 2D scatter plot with matplotlib."""
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and assign colors
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Plot each category
    for label in unique_labels:
        mask = [l == label for l in labels]
        indices = [i for i, m in enumerate(mask) if m]
        x = reduced_embeddings[indices, 0]
        y = reduced_embeddings[indices, 1]
        plt.scatter(x, y, c=[color_map[label]], label=label, s=100, alpha=0.7)
    
    # Add document labels (shortened)
    for i, doc in enumerate(documents):
        short_doc = doc[:25] + "..." if len(doc) > 25 else doc
        plt.annotate(
            short_doc,
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/embeddings_2d.png', dpi=150, bbox_inches='tight')
    print("\nSaved 2D plot to: output/embeddings_2d.png")
    plt.show()


def plot_3d_plotly(reduced_embeddings, labels, documents, title="Embeddings 3D"):
    """Create interactive 3D plot with Plotly."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping 3D plot.")
        return
    
    # Create hover text
    hover_texts = [f"{l}: {d[:50]}..." for l, d in zip(labels, documents)]
    
    fig = px.scatter_3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        color=labels,
        hover_name=hover_texts,
        title=title,
        labels={'x': 'Dim 1', 'y': 'Dim 2', 'z': 'Dim 3'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        width=900,
        height=700
    )
    
    # Save as HTML
    os.makedirs('output', exist_ok=True)
    fig.write_html('output/embeddings_3d.html')
    print("\nSaved 3D plot to: output/embeddings_3d.html")
    fig.show()


def plot_similarity_heatmap(embeddings, documents):
    """Create similarity heatmap."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(embeddings)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    
    # Add labels
    short_docs = [d[:20] + "..." for d in documents]
    plt.xticks(range(len(documents)), short_docs, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(documents)), short_docs, fontsize=8)
    
    plt.title('Document Similarity Heatmap')
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/similarity_heatmap.png', dpi=150, bbox_inches='tight')
    print("\nSaved heatmap to: output/similarity_heatmap.png")
    plt.show()


def main():
    print("="*60)
    print("EMBEDDING VISUALIZATION DEMO")
    print("="*60)
    
    # Create sample data
    documents, labels = create_sample_data()
    print(f"\nCreated {len(documents)} sample documents in {len(set(labels))} categories")
    
    # Generate embeddings
    embeddings = generate_embeddings(documents)
    
    # 2D Visualization with t-SNE
    print("\n" + "-"*40)
    print("2D VISUALIZATION (t-SNE)")
    print("-"*40)
    reduced_2d = reduce_dimensions_tsne(embeddings, n_components=2)
    plot_2d_matplotlib(reduced_2d, labels, documents, "Document Embeddings (t-SNE 2D)")
    
    # 3D Visualization with t-SNE
    print("\n" + "-"*40)
    print("3D VISUALIZATION (t-SNE)")
    print("-"*40)
    reduced_3d = reduce_dimensions_tsne(embeddings, n_components=3)
    plot_3d_plotly(reduced_3d, labels, documents, "Document Embeddings (t-SNE 3D)")
    
    # Similarity Heatmap
    print("\n" + "-"*40)
    print("SIMILARITY HEATMAP")
    print("-"*40)
    plot_similarity_heatmap(embeddings, documents)
    
    print("\n" + "="*60)
    print("Visualization complete! Check the 'output' folder.")
    print("="*60)


if __name__ == "__main__":
    main()
