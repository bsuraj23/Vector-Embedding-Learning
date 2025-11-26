# Vector Embedding Learning

A beginner-friendly guide to Vector Embeddings with hands-on examples: Document analysis, MySQL data embeddings, and 2D/3D visualizations using Python.

## What are Vector Embeddings?

Vector embeddings are numerical representations of data (text, images, audio, etc.) in a multi-dimensional space. They convert complex data into arrays of numbers that capture semantic meaning, allowing machines to understand relationships between concepts.

### Simple Analogy

Imagine organizing books in a library:
- Traditional approach: Alphabetically by title (no semantic understanding)
- Embedding approach: By topic similarity (Machine Learning books near AI books, both near Computer Science)

### Why are Embeddings Important?

| Feature | Traditional Search | Embedding-based Search |
|---------|-------------------|------------------------|
| "dog" matches "dog" | Yes | Yes |
| "dog" matches "puppy" | No | Yes (semantic similarity) |
| "dog" matches "canine" | No | Yes (semantic similarity) |
| Language understanding | Keyword only | Contextual meaning |

## Project Structure

```
Vector-Embedding-Learning/
|-- README.md
|-- requirements.txt
|-- 01_document_embeddings.py      # Text/Document analysis
|-- 02_mysql_embeddings.py         # MySQL table data embeddings
|-- 03_embedding_visualization.py  # 2D/3D visualization
|-- sample_data/
|   |-- sample_documents.txt
|   |-- mysql_setup.sql
|-- output/
|   |-- embeddings_2d.png
|   |-- embeddings_3d.html
```

## Installation

```bash
# Clone the repository
git clone https://github.com/bsuraj23/Vector-Embedding-Learning.git
cd Vector-Embedding-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start Examples

### 1. Document Embeddings

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Python is a popular programming language",
    "AI and ML are transforming industries"
]

# Generate embeddings
embeddings = model.encode(documents)
print(f"Embedding shape: {embeddings.shape}")  # (4, 384)
```

### 2. MySQL Data Embeddings

```python
import mysql.connector
from sentence_transformers import SentenceTransformer

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="your_user",
    password="your_password",
    database="your_database"
)

# Fetch data from table
cursor = conn.cursor()
cursor.execute("SELECT id, description FROM products")
rows = cursor.fetchall()

# Generate embeddings for each row
model = SentenceTransformer('all-MiniLM-L6-v2')
for row_id, description in rows:
    embedding = model.encode(description)
    print(f"ID: {row_id}, Embedding dim: {len(embedding)}")
```

### 3. Visualize Embeddings

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Reduce 384-dim embeddings to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, doc in enumerate(documents):
    plt.annotate(doc[:30] + "...", (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.title("Document Embeddings Visualization (2D)")
plt.savefig("output/embeddings_2d.png")
plt.show()
```

## Key Concepts

### Embedding Dimensions

- **all-MiniLM-L6-v2**: 384 dimensions (fast, lightweight)
- **all-mpnet-base-v2**: 768 dimensions (higher quality)
- **OpenAI text-embedding-ada-002**: 1536 dimensions

### Similarity Metrics

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Calculate similarity between two embeddings
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity: {similarity:.4f}")  # 0.0 to 1.0
```

### Common Use Cases

1. **Semantic Search**: Find documents by meaning, not just keywords
2. **Recommendation Systems**: Suggest similar items
3. **Clustering**: Group similar documents together
4. **RAG (Retrieval Augmented Generation)**: Enhance LLMs with relevant context
5. **Anomaly Detection**: Find outliers in data

## Dependencies

- sentence-transformers
- numpy
- matplotlib
- scikit-learn
- plotly (for 3D visualization)
- mysql-connector-python

## Learning Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Understanding Word Embeddings](https://jalammar.github.io/illustrated-word2vec/)
- [ChromaDB for Vector Storage](https://www.trychroma.com/)

## License

MIT License - See LICENSE file for details.

## Author

Created for learning purposes. Feel free to contribute!

---

**Star this repo if you find it helpful!**
