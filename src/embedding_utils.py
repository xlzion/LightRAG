# src/embedding_utils.py
from sentence_transformers import SentenceTransformer
import torch # [18]

# Global variable to cache the model
_embedding_model = None

def get_embedding_model(model_name: str, device: str = None):
    """Loads and returns a Sentence Transformer model. Caches the model."""
    global _embedding_model
    if _embedding_model is None:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model {model_name} on device: {device}")
        _embedding_model = SentenceTransformer(model_name, device=device)
        # For further optimization, consider quantization (e.g., INT8) using tools like Optimum
        # This typically involves converting the model to ONNX and then quantizing.
        # Example (conceptual, requires `optimum` library and setup):
        # from optimum.onnxruntime import ORTQuantizer, ORTModelForFeatureExtraction
        # from optimum.onnxruntime.configuration import AutoQuantizationConfig
        # onnx_path = "path_to_onnx_model"
        # _embedding_model.save_pretrained(onnx_path) # If model can be saved in a way Optimum understands
        # ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
        # quantizer = ORTQuantizer.from_pretrained(ort_model)
        # qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        # quantizer.quantize(save_dir="quantized_model", quantization_config=qconfig)
        # _embedding_model = SentenceTransformer("quantized_model", device=device)
        # Sentence Transformers also has some built-in utilities for quantization [1, 2, 3, 23, 24]
        # e.g., model.half() for FP16 if on GPU, or using sentence_transformers.util.quantize_embeddings
        if device == 'cuda' and hasattr(_embedding_model, 'half'):
             print("Using FP16 for embedding model on GPU.")
             _embedding_model.half() # Use FP16 for faster inference on compatible GPUs

    return _embedding_model

def generate_embeddings_with_cache(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Generates embeddings for a list of texts, using a cache.
    Optimized for when 'texts' is typically a single user query.
    """
    global _embedding_cache

    if not texts:
        return []

    # Assumes 'texts' for user queries will be a list with one item: [semantic_query]
    query_text = texts[0] 

    if query_text in _embedding_cache:
        print(f"Embedding for query found in cache: '{query_text[:100]}...'")
        return [_embedding_cache[query_text]] # Return as list of lists
    else:
        print(f"Embedding for query NOT in cache. Generating for: '{query_text[:100]}...'")
        new_embedding_np = model.encode([query_text], batch_size=1, show_progress_bar=False)
        new_embedding_list = new_embedding_np[0].tolist() # Get the 1D list

        _embedding_cache[query_text] = new_embedding_list # Cache the 1D list
        print(f"Cached new embedding for: '{query_text[:100]}...'")
        return [new_embedding_list] # Return as list of lists

def generate_embeddings(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generates embeddings for a list of texts."""
    print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
    # Ensure show_progress_bar is True for long ingestion, False for single query if desired
    show_progress = len(texts) > 1 
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress)
    return embeddings.tolist()


if __name__ == '__main__':
    from config import EMBEDDING_MODEL_NAME
    model = get_embedding_model(EMBEDDING_MODEL_NAME)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world.",
        "Natural Language Processing enables machines to understand human language."
    ]
    print("Sample texts for embedding generation:", sample_texts)
    print("Generating embeddings...")
    embeddings = generate_embeddings(model, sample_texts)
    print("\nGenerated embeddings shape:", (len(embeddings), len(embeddings) if embeddings else 0))
    # print("First embedding:", embeddings[:5]) # Print first 5 dims of first embedding