"""MySQL Data Embeddings - Beginner's Guide

This script demonstrates how to:
1. Connect to a MySQL database
2. Fetch data from tables
3. Generate embeddings for table data
4. Store embeddings back to database or use for similarity search

Author: Vector Embedding Learning Project
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# MySQL connector - install with: pip install mysql-connector-python
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("MySQL connector not installed. Using demo mode with sample data.")


class MySQLEmbeddings:
    """Class to handle MySQL data embeddings."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with embedding model."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.connection = None
        print("Model loaded successfully!")
    
    def connect_to_mysql(self, host, user, password, database):
        """Connect to MySQL database.
        
        Args:
            host: MySQL server host
            user: MySQL username
            password: MySQL password
            database: Database name
        """
        if not MYSQL_AVAILABLE:
            print("MySQL connector not available. Cannot connect.")
            return False
        
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            print(f"Connected to MySQL database: {database}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def fetch_table_data(self, table_name, text_columns, id_column='id'):
        """Fetch data from a MySQL table.
        
        Args:
            table_name: Name of the table
            text_columns: List of column names to embed
            id_column: Primary key column name
        
        Returns:
            List of tuples (id, combined_text)
        """
        if not self.connection:
            print("Not connected to database.")
            return []
        
        cursor = self.connection.cursor()
        columns = ', '.join([id_column] + text_columns)
        query = f"SELECT {columns} FROM {table_name}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        
        # Combine text columns into single text for embedding
        data = []
        for row in rows:
            row_id = row[0]
            combined_text = ' '.join([str(val) for val in row[1:] if val])
            data.append((row_id, combined_text))
        
        print(f"Fetched {len(data)} rows from {table_name}")
        return data
    
    def generate_embeddings(self, texts):
        """Generate embeddings for texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def find_similar_rows(self, query, data, embeddings, top_k=5):
        """Find rows most similar to a query.
        
        Args:
            query: Search query text
            data: List of (id, text) tuples
            embeddings: Pre-computed embeddings
            top_k: Number of results
        """
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                'id': data[idx][0],
                'text': data[idx][1],
                'similarity': float(similarities[idx])
            })
        return results
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")


def demo_with_sample_data():
    """Demo using sample data (no MySQL required)."""
    print("\n" + "="*60)
    print("DEMO MODE: Using sample product data")
    print("="*60 + "\n")
    
    # Sample data simulating a products table
    sample_data = [
        (1, "iPhone 15 Pro Max 256GB smartphone with A17 chip titanium design"),
        (2, "Samsung Galaxy S24 Ultra 512GB Android phone with AI features"),
        (3, "MacBook Pro 16 inch M3 Max chip 32GB RAM laptop for professionals"),
        (4, "Dell XPS 15 Intel i9 32GB RAM Windows laptop ultrabook"),
        (5, "Sony WH-1000XM5 wireless noise cancelling headphones premium audio"),
        (6, "Apple AirPods Pro 2nd generation wireless earbuds active noise cancellation"),
        (7, "iPad Pro 12.9 inch M2 chip tablet with Apple Pencil support"),
        (8, "Samsung Galaxy Tab S9 Ultra Android tablet with S Pen"),
        (9, "Canon EOS R5 mirrorless camera 45MP full frame professional photography"),
        (10, "Sony A7 IV mirrorless camera 33MP hybrid autofocus video creator"),
    ]
    
    # Initialize embeddings handler
    handler = MySQLEmbeddings()
    
    # Extract texts and generate embeddings
    ids = [row[0] for row in sample_data]
    texts = [row[1] for row in sample_data]
    
    print("\nGenerating embeddings for product data...")
    embeddings = handler.generate_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Semantic search examples
    print("\n" + "-"*60)
    print("SEMANTIC SEARCH DEMO")
    print("-"*60)
    
    queries = [
        "best phone for photography",
        "laptop for coding and development",
        "wireless earphones with good battery",
        "tablet for drawing and design"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("Top 3 matching products:")
        
        results = handler.find_similar_rows(query, sample_data, embeddings, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [ID: {result['id']}] (Score: {result['similarity']:.4f})")
            print(f"     {result['text'][:60]}...")
    
    return embeddings, sample_data


def main_mysql_example():
    """Example with real MySQL database."""
    print("\n" + "="*60)
    print("MySQL Database Example")
    print("="*60 + "\n")
    
    handler = MySQLEmbeddings()
    
    # Connect to MySQL (update with your credentials)
    # connected = handler.connect_to_mysql(
    #     host="localhost",
    #     user="your_username",
    #     password="your_password",
    #     database="your_database"
    # )
    # 
    # if connected:
    #     # Fetch data from products table
    #     data = handler.fetch_table_data(
    #         table_name="products",
    #         text_columns=["name", "description"],
    #         id_column="product_id"
    #     )
    #     
    #     # Generate embeddings
    #     texts = [row[1] for row in data]
    #     embeddings = handler.generate_embeddings(texts)
    #     
    #     # Search example
    #     results = handler.find_similar_rows(
    #         query="wireless headphones",
    #         data=data,
    #         embeddings=embeddings,
    #         top_k=5
    #     )
    #     
    #     for result in results:
    #         print(f"ID: {result['id']}, Score: {result['similarity']:.4f}")
    #     
    #     handler.close()
    
    print("MySQL example code is commented out.")
    print("Uncomment and update credentials to use with your database.")


if __name__ == "__main__":
    # Run demo with sample data
    embeddings, data = demo_with_sample_data()
    
    # Show how to use with MySQL
    print("\n" + "="*60)
    print("To use with MySQL, see main_mysql_example() function.")
    print("="*60)
