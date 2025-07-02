# src/vector_node_manager.py
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, UpdateStatus
from typing import List, Optional
import warnings
import os # For checking path
import shutil # For deleting path if needed in a more robust reset

from config import NODE_COLLECTION_NAME, EMBEDDING_DIMENSION, QDRANT_NODE_DB_PATH, NODE_VECTOR_NAME

_qdrant_client_instance_map = {} # Use a dictionary to manage instances per path

class VectorNodeManager:
    def __init__(self, path: str = QDRANT_NODE_DB_PATH, collection_name: str = NODE_COLLECTION_NAME,
                 vector_name: str = NODE_VECTOR_NAME, vector_size: int = EMBEDDING_DIMENSION):
        self.path = path
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.vector_size = vector_size
        self.models = models
        self.EXPLICIT_VECTOR_NAME = vector_name
        self.NODE_COLLECTION_NAME = collection_name
        self.client = self._get_or_create_client(force_new=False) # Get or create client on init
        self._create_collection_if_not_exists()

    def _get_or_create_client(self, force_new: bool = False) -> QdrantClient:
        global _qdrant_client_instance_map

        if force_new and self.path in _qdrant_client_instance_map:
            print(f"Forcing new Qdrant client instance for path: {self.path}")
            # Attempt to close existing client if it has a close method
            # Note: QdrantClient in local mode doesn't have an explicit 'close()' in older versions.
            # For newer versions (check docs if you upgrade), it might.
            # The lock is released when the client object is garbage collected or process ends.
            del _qdrant_client_instance_map[self.path] # Remove from map
            # Consider if the actual lock file needs more aggressive handling,
            # but usually deleting the instance reference should suffice if no other process holds it.

        if self.path not in _qdrant_client_instance_map:
            print(f"Initializing new Qdrant client instance for path: {self.path}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Local mode is not recommended.*", category=UserWarning)
                # If forcing new and path was just deleted, this creates a fresh client
                _qdrant_client_instance_map[self.path] = QdrantClient(path=self.path)

        return _qdrant_client_instance_map[self.path]

    def force_new_client_instance(self):
        """Forces re-initialization of the client for this manager's path."""
        print(f"VectorNodeManager: force_new_client_instance called for path {self.path}.")
        # Deleting the path BEFORE trying to get a new client is key if we want to ensure no lock.
        # This should be handled by main_cli.py's --force-rebuild-graph logic.
        # This method will now just ensure the *Python object* is new.
        self.client = self._get_or_create_client(force_new=True)
        self._create_collection_if_not_exists() # Re-ensure collection exists with the new client

    def _create_collection_if_not_exists(self): # This method uses self.client
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print(f"Node embedding collection '{self.collection_name}' already exists.")
        except Exception as e:
            # More specific exception handling could be added based on qdrant_client version
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower() or "status_code=404" in str(e).lower():
                print(f"Node embedding collection '{self.collection_name}' not found. Creating collection.")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.vector_name: models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE
                        )
                    }
                )
                print(f"Node embedding collection '{self.collection_name}' created with vector '{self.vector_name}'.")
                self.client.create_payload_index(collection_name=self.collection_name, field_name="node_type", field_schema="keyword")
                self.client.create_payload_index(collection_name=self.collection_name, field_name="name", field_schema="text")
                print(f"Payload indexes created.")
            else:
                print(f"Error checking/creating collection '{self.collection_name}': {e}")
                raise # Re-raise if it's not a simple "not found" error

    def search_nodes(self, query_embedding: List[float], 
                     filters: Optional[models.Filter] = None, 
                     top_k: int = 5) -> List[models.ScoredPoint]:
        print(f"Searching nodes in '{self.collection_name}' against vector '{self.vector_name}'.")
        try:
            search_params_obj = models.SearchParams(hnsw_ef=128, exact=False)
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=(self.vector_name, query_embedding),
                query_filter=filters,
                search_params=search_params_obj,
                limit=top_k,
                with_payload=True
            )
            print(f"Node search returned {len(search_result)} results.")
            return search_result
        except Exception as e:
            print(f"Error during node search in Qdrant: {e}")
            # Add diagnostics if needed
            return []

# Note: Ingestion of node embeddings is now handled by `build_graph_from_elements` in graph_db_manager.py
# if it's passed the embedding_model and an instance of VectorNodeManager.