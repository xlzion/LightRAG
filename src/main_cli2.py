# src/main_cli.py
import argparse
import time
import os
import shutil

from config import (
    EMBEDDING_MODEL_NAME, NODE_COLLECTION_NAME, EMBEDDING_DIMENSION, XLSX_FILE_PATH,
    QDRANT_NODE_DB_PATH, GRAPH_FILE_PATH, LLM_MODEL_IDENTIFIER, NODE_VECTOR_NAME
)
from embedding_utils import get_embedding_model
from data_processor_graph import load_book_data, extract_graph_elements
from graph_db_manager import get_graph, build_graph_from_elements, save_graph
from vector_node_manager import VectorNodeManager
from rag_pipeline_graph import process_graph_query

def setup_graph_rag_system(force_rebuild_graph_flag=False):
    """Initializes models, DB, graph, and ingests data if needed."""
    print("Setting up GraphRAG system...")
    start_time = time.time()

    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)

    if force_rebuild_graph_flag:
        print("--force-rebuild-graph is True. Deleting existing graph file and Qdrant node DB...")
        if os.path.exists(GRAPH_FILE_PATH):
            os.remove(GRAPH_FILE_PATH)
            print(f"Deleted old graph file: {GRAPH_FILE_PATH}")
        if os.path.exists(QDRANT_NODE_DB_PATH):
            shutil.rmtree(QDRANT_NODE_DB_PATH)
            print(f"Deleted old Qdrant Node DB: {QDRANT_NODE_DB_PATH}")
    
    qdrant_parent_dir = os.path.dirname(QDRANT_NODE_DB_PATH)
    if not os.path.exists(qdrant_parent_dir):
        os.makedirs(qdrant_parent_dir)
    graph_parent_dir = os.path.dirname(GRAPH_FILE_PATH)
    if not os.path.exists(graph_parent_dir):
        os.makedirs(graph_parent_dir)

    vector_node_mgr = VectorNodeManager(
        path=QDRANT_NODE_DB_PATH,
        collection_name=NODE_COLLECTION_NAME,
        vector_name=NODE_VECTOR_NAME, # Ensure NODE_VECTOR_NAME is defined in config or here
        vector_size=EMBEDDING_DIMENSION
    )

    graph = get_graph(graph_path=GRAPH_FILE_PATH, force_reload=force_rebuild_graph_flag)

    build_is_needed = False
    if force_rebuild_graph_flag:
        build_is_needed = True
        print("Rebuilding graph and re-ingesting node embeddings due to --force-rebuild-graph.")
    else:
        if graph.number_of_nodes() == 0:
            build_is_needed = True
            print("Graph is empty (file not found or empty). Building graph and ingesting node embeddings.")
        else:
            print(f"Graph loaded with {graph.number_of_nodes()} nodes. Checking Qdrant node embeddings...")
            try:
                collection_info = vector_node_mgr.client.get_collection(vector_node_mgr.collection_name)
                if not (hasattr(collection_info, 'points_count') and collection_info.points_count > 0 and collection_info.points_count >= graph.number_of_nodes() * 0.9): # Check if roughly enough points exist
                    build_is_needed = True
                    print(f"Qdrant collection '{vector_node_mgr.collection_name}' appears empty or incomplete. Re-ingesting node embeddings.")
                else:
                    print(f"Qdrant collection '{vector_node_mgr.collection_name}' has {collection_info.points_count} points. Skipping ingestion.")
            except Exception as e:
                print(f"Could not get Qdrant collection info (error: {e}). Assuming node embeddings need ingestion.")
                build_is_needed = True

    if build_is_needed:
        print("Proceeding with graph building and/or node embedding ingestion...")
        book_entries = load_book_data(XLSX_FILE_PATH)
        if book_entries:
            nodes_data, edges_data = extract_graph_elements(book_entries)
            graph = build_graph_from_elements(nodes_data, edges_data,
                                              embedding_model=embedding_model,
                                              vector_node_manager=vector_node_mgr)
            save_graph(graph, GRAPH_FILE_PATH)
            print("Graph built/updated and node embeddings ingested/updated.")
        else:
            print("CRITICAL ERROR: No book data found from XLSX. Cannot build graph or ingest embeddings.")
            return None, None, None, None
    else:
        print(f"Graph already loaded ({graph.number_of_nodes()} nodes) and Qdrant node embeddings exist. Skipping build/ingestion.")

    end_time = time.time()
    print(f"GraphRAG system setup completed in {end_time - start_time:.2f} seconds.")
    # Return LLM_MODEL_IDENTIFIER from config for use in process_graph_query
    return embedding_model, vector_node_mgr, graph, LLM_MODEL_IDENTIFIER

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Digital Librarian CLI - Conversational Mode")
    parser.add_argument("--force-rebuild-graph", action="store_true", 
                        help="Force complete rebuild of the graph from XLSX and re-ingestion of all node embeddings.")
    parser.add_argument("--query", type=str, help="Optional: Ask an initial question and then exit (non-conversational).")

    args = parser.parse_args()

    print("Initializing GraphRAG system...")
    # Get all necessary components from setup, including the llm_model_id
    components = setup_graph_rag_system(
        force_rebuild_graph_flag=args.force_rebuild_graph
    )
    
    if not components or not all(c is not None for c in components):
        print("Failed to initialize GraphRAG components properly. Exiting.")
        return
        
    embedding_model, vector_node_mgr, graph, llm_model_id = components


    if not graph or (graph.number_of_nodes() == 0 and not args.force_rebuild_graph):
        print("Graph is empty and not rebuilding. Exiting.")
        return
    
    if args.query: # If an initial query is provided via command line, process it and exit
        print(f"\nProcessing initial query: '{args.query}'")
        start_query_time = time.time()
        response = process_graph_query(args.query, embedding_model, vector_node_mgr, graph, llm_model_id)
        end_query_time = time.time()
        print("\n--- Digital Librarian Response ---")
        print(response)
        print("--- End of Response ---")
        print(f"\nQuery processed in {end_query_time - start_query_time:.2f} seconds.")
    else: # Enter conversational mode
        print("\n--- Digital Librarian Conversational Mode ---")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Hello! I'm the Digital Librarian in the CityUHK(DG). Feel free to ask me anything about the library's collection or services s.")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting Digital Librarian. Goodbye!")
                    break
                if not user_input: # Skip empty input
                    continue

                start_query_time = time.time()
                response = process_graph_query(user_input, embedding_model, vector_node_mgr, graph, llm_model_id)
                end_query_time = time.time()

                print("\nLibrarian:")
                print(response)
                print(f"(Processed in {end_query_time - start_query_time:.2f} seconds)\n")

            except KeyboardInterrupt:
                print("\nExiting Digital Librarian due to interrupt. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred during conversation: {e}")
                # Depending on error severity, you might want to break or allow continuation
                # For now, it continues the loop.
                # break

if __name__ == "__main__":
    main()