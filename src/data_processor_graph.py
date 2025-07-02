# src/data_processor_graph.py
import pandas as pd
import re

def load_book_data(xlsx_path: str) -> list[dict]:
    try:
        df = pd.read_excel(xlsx_path)
        df = df.fillna('')
        return df.to_dict('records')
    except FileNotFoundError:
        print(f"Error: XLSX file not found at {xlsx_path}")
        return []
    except Exception as e:
        print(f"Error loading XLSX file: {e}")
        return []

def extract_graph_elements(book_entries: list[dict]) -> (list, list):
    """
    Extracts nodes (books, authors) and relationships from book entries.
    Returns:
        nodes_data: list of tuples, e.g., (node_id, node_type, attributes_dict)
        edges_data: list of tuples, e.g., (source_node_id, target_node_id, relationship_type)
    """
    nodes_data = []
    edges_data = []
    authors_set = {} # To store unique authors and assign them IDs

    for idx, entry in enumerate(book_entries):
        book_id = f"book_{idx}" # Simple unique ID for the book
        book_title = entry.get('Book Name', '').strip()
        raw_author_string = entry.get('Author', '').strip()
        
        # Basic author parsing (can be improved)
        # This example assumes authors might be listed together and tries to split them if common patterns exist.
        # A more robust solution would require better parsing logic or cleaner data.
        current_book_author_ids = []
        # Simplistic split, assumes authors separated by common delimiters like ';', '/', or ' and '
        # This part needs careful adjustment based on your actual 'Author' field format.
        # For "Kalita, Jugal Kumar, (London, ...)", a more complex regex might be needed to get just "Kalita, Jugal Kumar"
        # For now, let's assume a simpler structure or pre-cleaned author names.
        # A simple approach: treat the whole author string as one author name for now if parsing is complex.
        # A more robust approach would be to pre-process the author field in the XLSX.
        
        # Example: if Author field contains "Jugal K. Kalita, Dhruba K. Bhattacharyya"
        # For this example, let's assume a single primary author name can be extracted or is the main one.
        # This parsing needs to be robust for your specific data.
        # Let's assume for now `raw_author_string` is the primary author name or a parsable list.
        
        # Simplified author handling: Treat the raw string as one author name for now
        # More sophisticated parsing would be needed for multi-author strings.
        author_name = raw_author_string
        if not author_name: # Handle empty author strings
            author_name = "Unknown Author"

        if author_name not in authors_set:
            author_id = f"author_{len(authors_set)}"
            authors_set[author_name] = author_id
            nodes_data.append((author_id, "author", {"name": author_name}))
        else:
            author_id = authors_set[author_name]
        
        current_book_author_ids.append(author_id)

        book_attributes = {
            "title": book_title,
            "full_author_string": raw_author_string, # Store original author string
            "storage_location": entry.get('Storage Location', ''),
            "call_number": entry.get('Call Number', ''),
            # Store the text that will be used for this book node's embedding
            "text_for_embedding": f"Title: {book_title}. Author: {author_name}." # Use the processed primary author name
        }
        nodes_data.append((book_id, "book", book_attributes))

        for auth_id in current_book_author_ids:
            edges_data.append((book_id, auth_id, "WRITTEN_BY"))
            edges_data.append((auth_id, book_id, "WROTE")) # Bidirectional for easier traversal

    return nodes_data, edges_data

if __name__ == '__main__':
    from config import XLSX_FILE_PATH
    books = load_book_data(XLSX_FILE_PATH)
    if books:
        print(f"Loaded {len(books)} book entries.")
        nodes, edges = extract_graph_elements(books[:5]) # Test with first 5
        print("\nSample Nodes:")
        for node in nodes:
            print(node)
        print("\nSample Edges:")
        for edge in edges:
            print(edge)