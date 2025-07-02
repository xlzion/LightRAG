# src/config.py
import os

# File Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
XLSX_FILE_PATH = os.path.join(DATA_DIR, "library_books.xlsx")
GRAPH_FILE_PATH = os.path.join(DATA_DIR, "library_graph.graphml") # For NetworkX graph

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Vector Database (Qdrant for Node Embeddings)
QDRANT_NODE_DB_PATH = os.path.join(DATA_DIR, "qdrant_node_db")
NODE_COLLECTION_NAME = "library_node_embeddings_collection"
NODE_VECTOR_NAME = "node_embedding_vector" # Explicit name for node vectors

# LLM Configuration
LLM_API_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL_IDENTIFIER = "qwen2.5:7b-instruct" # Or your chosen model

# Search/Retrieval parameters
TOP_K_NODE_SEARCH = 5 # How many initial nodes to retrieve from vector search
GRAPH_TRAVERSAL_DEPTH = 2 # Example depth for graph exploration
TOP_K_RESULTS_FOR_LLM = 3 # How many final items to send to LLM