# src/rag_pipeline_graph.py
import re
from sentence_transformers import SentenceTransformer # For type hinting
from typing import Optional

from config import TOP_K_NODE_SEARCH, GRAPH_TRAVERSAL_DEPTH, TOP_K_RESULTS_FOR_LLM, LLM_MODEL_IDENTIFIER
from embedding_utils import generate_embeddings_with_cache ,generate_embeddings # Assuming you use the caching version for queries
from graph_db_manager import get_graph, find_relevant_nodes_and_neighbors, nx # Import nx
from vector_node_manager import VectorNodeManager # Import the class
from llm_handler import generate_llm_response
from qdrant_client import models # For filter creation

def extract_filters_from_query_graph(query: str) -> (str, Optional[models.Filter], list[str]):
    """
    Extracts filters for vector search (e.g., node_type='author') and potential direct graph entity lookups.
    Returns: semantic_query, qdrant_filter_for_nodes, list_of_direct_entity_names
    """
    author_match = re.search(r"author:\s*([\w\s\.\-]+)", query, re.IGNORECASE) # Adjusted regex for names
    qdrant_filter_conditions = []
    direct_entity_names = [] # For direct graph lookup if author name is explicitly given

    if author_match:
        author_name = author_match.group(1).strip()
        query = query.replace(author_match.group(0), "").strip()
        # For Qdrant, filter where node_type is author and name matches
        qdrant_filter_conditions.append(
            models.FieldCondition(key="node_type", match=models.MatchValue(value="author"))
        )
        qdrant_filter_conditions.append(
            models.FieldCondition(key="name", match=models.MatchText(text=author_name)) # Use MatchText for name
        )
        direct_entity_names.append(author_name) # Add for direct graph lookup as well
        print(f"Extracted author entity/filter: {author_name}")
    
    # You might add other entity extractions here (e.g., book titles if quoted)

    if qdrant_filter_conditions:
        return query.strip(), models.Filter(must=qdrant_filter_conditions), direct_entity_names
    return query.strip(), None, direct_entity_names

def linearize_graph_context_for_llm(retrieved_nodes_data: list[dict], graph: nx.Graph) -> str:
    """Converts retrieved graph nodes and their immediate context into a string for the LLM."""
    if not retrieved_nodes_data:
        return "No specific information found in the graph."

    context_parts = []
    # Limit the number of nodes processed for the context to avoid overly long prompts
    for node_data in retrieved_nodes_data[:TOP_K_RESULTS_FOR_LLM]: # Use a config for this limit
        node_id = node_data.get('id')
        node_type = node_data.get('type')
        
        if node_type == "book":
            title = node_data.get('title', 'N/A')
            authors = []
            # Find authors connected to this book in the graph
            if graph.has_node(node_id):
                for neighbor_id in graph.neighbors(node_id):
                    if graph.nodes[neighbor_id].get('type') == 'author':
                        authors.append(graph.nodes[neighbor_id].get('name', 'Unknown Author'))
            author_str = ", ".join(authors) if authors else node_data.get('full_author_string', 'N/A') # Fallback to original string
            location = node_data.get('storage_location', 'N/A')
            call_number = node_data.get('call_number', 'N/A')
            context_parts.append(
                f"- Book: '{title}' by {author_str}. Location: {location}, Call#: {call_number}."
            )
        elif node_type == "author":
            name = node_data.get('name', 'N/A')
            # Find books written by this author
            books_by_author = []
            if graph.has_node(node_id):
                for neighbor_id in graph.neighbors(node_id):
                    if graph.nodes[neighbor_id].get('type') == 'book':
                        books_by_author.append(graph.nodes[neighbor_id].get('title', 'Unknown Title'))
            books_str = "; ".join(books_by_author[:3]) if books_by_author else "some works" # Limit books listed
            context_parts.append(
                f"- Author: {name}, known for works like '{books_str}'."
            )
        # Add more types if you have them (e.g., keywords)
        
    if not context_parts:
        return "Found some related items, but could not form detailed context."
        
    return "\n".join(context_parts)


def process_graph_query(
    query_text: str,
    embedding_model: SentenceTransformer,
    vector_node_mgr: VectorNodeManager, 
    graph: nx.Graph, 
    llm_model_id: str = LLM_MODEL_IDENTIFIER
) -> str:
    print(f"\nProcessing graph query: '{query_text}'")

    semantic_query, qdrant_node_filters, direct_entity_names = extract_filters_from_query_graph(query_text)
    # ... (semantic query and filter processing) ...

    start_node_ids_from_graph = []
    if direct_entity_names:
        for entity_name in direct_entity_names:
            for node_id, data in graph.nodes(data=True):
                if data.get('name') == entity_name and data.get('type') == 'author':
                    start_node_ids_from_graph.append(node_id)
                    print(f"Found direct graph node ID '{node_id}' for entity '{entity_name}'.")
                    break 
    
    query_embedding_1d = [] # Ensure this is a 1D list
    if semantic_query:
        # generate_embeddings here will print "Generating embeddings for 1 texts..."
        list_of_embeddings = generate_embeddings(embedding_model, [semantic_query], batch_size=1) 
        if list_of_embeddings and list_of_embeddings[0]:
            query_embedding_1d = list_of_embeddings[0]
    
    retrieved_node_points_from_qdrant = []
    if query_embedding_1d:
        retrieved_node_points_from_qdrant = vector_node_mgr.search_nodes(
            query_embedding=query_embedding_1d,
            filters=qdrant_node_filters,
            top_k=TOP_K_NODE_SEARCH
        )
    
    start_node_ids_from_vector_search = []
    if retrieved_node_points_from_qdrant:
        print("--- Retrieved Qdrant Points (for graph node IDs) ---")
        for point in retrieved_node_points_from_qdrant:
            graph_node_id = point.payload.get('graph_node_id') # <--- USE THIS
            if graph_node_id:
                print(f"  Qdrant Point ID: {point.id}, Graph Node ID: {graph_node_id}, Score: {point.score:.4f}, Payload: {point.payload}")
                start_node_ids_from_vector_search.append(graph_node_id)
            else:
                print(f"  Warning: Qdrant Point ID {point.id} missing 'graph_node_id' in payload. Payload: {point.payload}")
        print("-----------------------------------------------------")
    
    combined_start_node_ids = list(set(start_node_ids_from_graph + start_node_ids_from_vector_search))
    print(f"Combined start graph node IDs for traversal: {combined_start_node_ids}")

    if not combined_start_node_ids:
        # If only filters were provided and no semantic query, or no vector search results,
        # you might still try to find nodes matching filters directly in the graph if not done already.
        # For now, this path assumes vector search or direct entity name provides start nodes.
        return "I couldn't find a strong starting point in our knowledge graph for your query."

    relevant_nodes_data = find_relevant_nodes_and_neighbors(graph, combined_start_node_ids, depth=GRAPH_TRAVERSAL_DEPTH)

    if not relevant_nodes_data:
        return "I found some initial matches but couldn't retrieve further details from the graph."

    context_string = linearize_graph_context_for_llm(relevant_nodes_data, graph)
    
    final_response = generate_llm_response(context_string, query_text, llm_model_id)
    return final_response