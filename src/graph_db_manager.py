# src/graph_db_manager.py
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional
import uuid # <--- IMPORT THE UUID MODULE
import os

from config import GRAPH_FILE_PATH
# Ensure correct import for SentenceTransformer if type hinting
# from embedding_utils import SentenceTransformer
# Use generate_embeddings (or your renamed batch version) from embedding_utils
from embedding_utils import generate_embeddings as generate_batch_node_embeddings


_graph = None

def get_graph(graph_path: str = GRAPH_FILE_PATH, force_reload: bool = False) -> nx.Graph:
    global _graph
    if _graph is None or force_reload:
        try:
            if force_reload and os.path.exists(graph_path): # Ensure deletion if forcing reload
                print(f"Force reload: Deleting existing graph file at {graph_path}")
                os.remove(graph_path)
            print(f"Loading graph from {graph_path}...")
            _graph = nx.read_graphml(graph_path)
            print(f"Graph loaded successfully: {_graph.number_of_nodes()} nodes, {_graph.number_of_edges()} edges.")
        except FileNotFoundError:
            print(f"Graph file not found at {graph_path}. Creating a new graph.")
            _graph = nx.Graph()
        except Exception as e:
            print(f"Error loading graph: {e}. Creating a new graph.")
            _graph = nx.Graph()
    return _graph

def save_graph(graph: nx.Graph, graph_path: str = GRAPH_FILE_PATH):
    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(graph_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
            print(f"Created directory for graph file: {parent_dir}")
            
        print(f"Saving graph to {graph_path}...")
        nx.write_graphml(graph, graph_path)
        print("Graph saved successfully.")
    except Exception as e:
        print(f"Error saving graph: {e}")

def build_graph_from_elements(nodes_data: List[Tuple[str, str, Dict]], 
                              edges_data: List[Tuple[str, str, str]],
                              embedding_model, # Pass the loaded SentenceTransformer model
                              vector_node_manager # Pass the VectorNodeManager instance
                              ):
    graph = nx.Graph()
    node_data_for_embedding_storage = [] 

    print("Building graph nodes...")
    for graph_node_id_str, node_type, attrs in nodes_data: # graph_node_id_str is "book_0", "author_0"
        graph.add_node(graph_node_id_str, type=node_type, **attrs)
        # Prepare data for embedding and Qdrant storage
        if "text_for_embedding" in attrs:
            node_data_for_embedding_storage.append({
                "graph_node_id": graph_node_id_str, 
                "text_to_embed": attrs["text_for_embedding"],
                "qdrant_payload": { 
                    "graph_node_id": graph_node_id_str, # Store original graph ID in payload
                    "node_type": node_type,
                    "name": attrs.get("title") or attrs.get("name", "") # For easy identification
                }
            })

    print("Building graph edges...")
    for source, target, rel_type in edges_data:
        if graph.has_node(source) and graph.has_node(target):
            graph.add_edge(source, target, type=rel_type)
        else:
            print(f"Warning: Skipping edge ({source}, {target}) due to missing node(s).")
    
    global _graph # Update the global graph instance if you use one
    _graph = graph

    if node_data_for_embedding_storage and embedding_model and vector_node_manager:
        print(f"Preparing to generate and store embeddings for {len(node_data_for_embedding_storage)} nodes...")
        
        texts_to_embed = [item["text_to_embed"] for item in node_data_for_embedding_storage]
        
        # Use your batch embedding function from embedding_utils.py
        # Assuming generate_batch_node_embeddings is an alias for generate_embeddings
        node_embeddings = generate_batch_node_embeddings(embedding_model, texts_to_embed, batch_size=32) 
        
        points_for_qdrant = []
        for i, item_data in enumerate(node_data_for_embedding_storage):
            # Generate a new, valid UUID string for Qdrant point ID
            qdrant_point_id = str(uuid.uuid4()) 
            
            points_for_qdrant.append(
                vector_node_manager.models.PointStruct(
                    id=qdrant_point_id, # Use the generated UUID string
                    vector={vector_node_manager.EXPLICIT_VECTOR_NAME: node_embeddings[i]},
                    payload=item_data["qdrant_payload"] # This payload contains the original graph_node_id
                )
            )
        
        if points_for_qdrant:
            print(f"Upserting {len(points_for_qdrant)} node embeddings to Qdrant...")
            vector_node_manager.client.upsert(
                collection_name=vector_node_manager.NODE_COLLECTION_NAME,
                points=points_for_qdrant,
                wait=True
            )
            print("Node embeddings stored in Qdrant.")
    return graph


def find_relevant_nodes_and_neighbors(graph: nx.Graph, 
                                      start_node_ids: List[str], 
                                      depth: int = 1) -> List[Dict]:
    # ... (this function remains the same, it operates on graph_node_ids) ...
    if not graph:
        return []
        
    all_relevant_nodes_data = []
    collected_node_ids = set()

    for start_node_id in start_node_ids:
        if start_node_id not in graph:
            print(f"Warning: Start node '{start_node_id}' not in graph for traversal.")
            continue

        if start_node_id not in collected_node_ids:
            start_node_data = dict(graph.nodes[start_node_id])
            start_node_data['id'] = start_node_id 
            all_relevant_nodes_data.append(start_node_data)
            collected_node_ids.add(start_node_id)

        queue = [(start_node_id, 0)]
        visited_for_traversal = {start_node_id}
        head = 0
        while head < len(queue):
            current_node_id, current_depth = queue[head]
            head += 1
            if current_depth < depth:
                for neighbor_id in graph.neighbors(current_node_id):
                    if neighbor_id not in visited_for_traversal:
                        visited_for_traversal.add(neighbor_id)
                        queue.append((neighbor_id, current_depth + 1))
                        if neighbor_id not in collected_node_ids:
                            neighbor_data = dict(graph.nodes[neighbor_id])
                            neighbor_data['id'] = neighbor_id
                            all_relevant_nodes_data.append(neighbor_data)
                            collected_node_ids.add(neighbor_id)
                            
    print(f"Graph traversal found {len(all_relevant_nodes_data)} relevant nodes.")
    return all_relevant_nodes_data


if __name__ == '__main__':
    from data_processor_graph import load_book_data, extract_graph_elements
    
    # Test graph creation and persistence
    # 1. Delete old graph file for a clean test if it exists
    #import os
    #if os.path.exists(GRAPH_FILE_PATH):
    #    os.remove(GRAPH_FILE_PATH)

    graph = get_graph(force_reload=True) # Get a new or loaded graph
    
    if graph.number_of_nodes() == 0: # Only build if graph is empty
        print("Building graph from XLSX...")
        book_entries = load_book_data() # Assumes XLSX_FILE_PATH from config
        if book_entries:
            nodes_data, edges_data = extract_graph_elements(book_entries)
            # For this direct test, we are not setting up Qdrant or embeddings
            graph = build_graph_from_elements(nodes_data, edges_data) 
            save_graph(graph)
        else:
            print("No book data to build graph.")
    else:
        print("Graph already loaded.")

    if graph.number_of_nodes() > 0:
        # Example traversal (assuming some nodes exist)
        sample_start_node_id = list(graph.nodes())[0] if list(graph.nodes()) else None
        if sample_start_node_id:
            print(f"\nTraversing from sample node: {sample_start_node_id}")
            relevant_data = find_relevant_nodes_and_neighbors(graph, [sample_start_node_id], depth=1)
            for item in relevant_data[:5]: # Print first 5
                print(item)