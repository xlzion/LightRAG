# src/main_cli.py
import argparse
import time
import os
import shutil

from config import (
    EMBEDDING_MODEL_NAME, NODE_COLLECTION_NAME, EMBEDDING_DIMENSION, XLSX_FILE_PATH,
    QDRANT_NODE_DB_PATH, GRAPH_FILE_PATH, LLM_MODEL_IDENTIFIER ,NODE_VECTOR_NAME
)
from embedding_utils import get_embedding_model
from data_processor_graph import load_book_data, extract_graph_elements
from graph_db_manager import get_graph, build_graph_from_elements, save_graph 
from vector_node_manager import VectorNodeManager 
from rag_pipeline_graph import process_graph_query 

def setup_graph_rag_system(force_rebuild_graph=False, force_reingest_nodes=False):
    print("Setting up GraphRAG system...")
    start_time = time.time()

    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)

    # Handle Qdrant DB deletion and path creation BEFORE client initialization
    if force_reingest_nodes: # This flag is tied to force_rebuild_graph in your main()
        if os.path.exists(QDRANT_NODE_DB_PATH):
            print(f"Force reingest nodes: Deleting existing Qdrant Node DB at {QDRANT_NODE_DB_PATH}...")
            shutil.rmtree(QDRANT_NODE_DB_PATH)
            print("Old Qdrant Node DB deleted.")

    # Ensure the parent data directory for Qdrant exists
    qdrant_parent_dir = os.path.dirname(QDRANT_NODE_DB_PATH)
    if not os.path.exists(qdrant_parent_dir):
        os.makedirs(qdrant_parent_dir) # Create data/ if not exists
        print(f"Created directory: {qdrant_parent_dir}")
    # If QDRANT_NODE_DB_PATH itself was deleted, QdrantClient will create it.

    # Initialize VectorNodeManager. It will create/get client and ensure collection.
    # If QDRANT_NODE_DB_PATH was just deleted, its internal _get_or_create_client
    # will make a new client for a new DB.
    vector_node_mgr = VectorNodeManager(
        path=QDRANT_NODE_DB_PATH,
        collection_name=NODE_COLLECTION_NAME,
        vector_name=NODE_VECTOR_NAME,
        vector_size=EMBEDDING_DIMENSION
    )
    # No need to call vector_node_mgr.force_new_client_instance() here if main_cli controls DB deletion.
    # The VectorNodeManager __init__ handles getting/creating the client.

    graph = get_graph(graph_path=GRAPH_FILE_PATH, force_reload=force_rebuild_graph)

    # Check if graph needs building or node embeddings need ingestion
    qdrant_collection_info = vector_node_mgr.client.get_collection(vector_node_mgr.collection_name)
    qdrant_points_exist = hasattr(qdrant_collection_info, 'points_count') and qdrant_collection_info.points_count > 0

    # Build graph and ingest node embeddings if graph is empty/forced OR if Qdrant is empty/forced
    if graph.number_of_nodes() == 0 or force_rebuild_graph or not qdrant_points_exist:
        print("Building graph and/or ingesting node embeddings...")
        # ... (rest of your graph building and node embedding ingestion logic using vector_node_mgr) ...
        # Ensure build_graph_from_elements uses the vector_node_mgr instance for upserting
        book_entries = load_book_data(XLSX_FILE_PATH)
        if book_entries:
            nodes_data, edges_data = extract_graph_elements(book_entries)
            graph = build_graph_from_elements(nodes_data, edges_data,
                                              embedding_model=embedding_model,
                                              vector_node_manager=vector_node_mgr) # Pass the manager
            save_graph(graph, GRAPH_FILE_PATH)
            print("Graph built and node embeddings ingested.")
        else:
            print("No book data to build graph or ingest node embeddings.")
    else:
        print("Graph loaded and node embeddings likely already exist.")
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        print(f"Qdrant node collection '{vector_node_mgr.collection_name}' has {qdrant_collection_info.points_count if qdrant_points_exist else 0} points.")

    end_time = time.time()
    print(f"GraphRAG system setup completed in {end_time - start_time:.2f} seconds.")
    return embedding_model, vector_node_mgr, graph

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Digital Librarian CLI")
    parser.add_argument("query", type=str, help="Your question for the digital librarian.")
    parser.add_argument("--force-rebuild-graph", action="store_true", help="Force rebuild of the graph from XLSX and re-ingest node embeddings.")
    # Renamed for clarity from --force-ingest
    args = parser.parse_args()

    # Determine if Qdrant node embeddings also need re-ingestion (usually if graph is rebuilt)
    force_reingest_node_embeddings = args.force_rebuild_graph 

    embedding_model, vector_node_mgr, graph = setup_graph_rag_system(
        force_rebuild_graph=args.force_rebuild_graph,
        force_reingest_nodes=force_reingest_node_embeddings 
    )

    if not embedding_model or not vector_node_mgr or not graph:
        print("Failed to initialize GraphRAG components. Exiting.")
        return
    
    print(f"\nAsking Digital Librarian (GraphRAG): '{args.query}'")
    start_query_time = time.time()
    response = process_graph_query(args.query, embedding_model, vector_node_mgr, graph, LLM_MODEL_IDENTIFIER)
    end_query_time = time.time()

    print("\n--- Digital Librarian Response ---")
    print(response)
    print("--- End of Response ---")
    print(f"\nQuery processed in {end_query_time - start_query_time:.2f} seconds.")

if __name__ == "__main__":
    main()