import numpy as np
from state import QueryState
import faiss
import pickle
 
def retriever(state: QueryState) -> QueryState:
    try:
        # Load FAISS index if not loaded
        if state.faiss_index is None:
            state.faiss_index = faiss.read_index(state.index_path)
 
        # Load metadata pickle if not loaded
        if state.loaded_metadatas is None:
            with open(state.metadata_path, "rb") as f:
                state.loaded_metadatas = pickle.load(f)
 
        texts = state.loaded_metadatas["texts"]
        metadatas = state.loaded_metadatas["metadatas"]
 
        query_lower = state.query.lower()
        # Determine retrieval strategy based on query
        k = 3  # Default: Only get top 3 most relevant chunks
        print(f"Query modality: {state.query_modality}")
        print(f"Query: {state.query}")
        # Apply filtering based on query content and modality
        if "table of contents" in query_lower or "contents" in query_lower:
            # Very specific - only get titles from first few pages
            filtered_idxs = [
                i for i, meta in enumerate(metadatas) 
                if (meta.get("chunk_type") == "text" and 
                    meta.get("element_category") == "Title" and
                    meta.get("page_numbers", [999])[0] <= 3)
            ]
            k = min(5, len(filtered_idxs))
        elif any(keyword in query_lower for keyword in ["when", "what", "how", "why", "definition", "define", "adopted", "regulation"]):
            # Factual questions - focus on text content, prefer narrative text and titles
            filtered_idxs = [
                i for i, meta in enumerate(metadatas) 
                if (meta.get("chunk_type") == "text" and 
                    meta.get("element_category") in ["Title", "NarrativeText"])
            ]
            k = 3
        elif state.query_modality == "table_required":
            # User explicitly wants tables
            filtered_idxs = [i for i, meta in enumerate(metadatas) if meta.get("chunk_type") == "table"]
            k = 5  # More tables if requested
        elif state.query_modality == "visual_required":
            # User explicitly wants visual content
            filtered_idxs = [i for i, meta in enumerate(metadatas) if meta.get("chunk_type") == "visual"]
            k = 3
        elif state.query_modality == "mixed":
            # Mixed content - no filtering
            filtered_idxs = list(range(len(metadatas)))
            k = 5
        else:  # text_only or default
            # Focus on text content
            filtered_idxs = [
                i for i, meta in enumerate(metadatas) 
                if meta.get("chunk_type") == "text"
            ]
            k = 3
 
        # If nothing matched, fall back but keep it small
        if not filtered_idxs:
            print("No filtered results, using all chunks")
            filtered_idxs = list(range(len(metadatas)))
            k = 2
 
        print(f"Filtered to {len(filtered_idxs)} chunks, will retrieve top {k}")
 
        # Prepare embeddings from FAISS index for filtered chunks
        all_embeddings = state.faiss_index.reconstruct_n(0, state.faiss_index.ntotal)
        embeddings = np.array([all_embeddings[i] for i in filtered_idxs])
 
        if embeddings.size == 0:
            state.retrieved_chunks = []
            return state
 
        # Build temporary FAISS index on filtered embeddings
        dim = embeddings.shape[1]
        tmp_index = faiss.IndexFlatL2(dim)
        tmp_index.add(embeddings)
 
        # Search with the determined k
        k = min(k, embeddings.shape[0])
        query_emb = np.array(state.query_embedding, dtype=np.float32).reshape(1, -1)
        D, I = tmp_index.search(query_emb, k=k)
 
        # Apply distance threshold - only keep very relevant results
        distance_threshold = 2.0  # Slightly more lenient for factual questions
        matched_chunks = []
        for idx, distance in zip(I[0], D[0]):
            if distance < distance_threshold:  # Only keep close matches
                original_idx = filtered_idxs[idx]
                matched_chunks.append({
                    "text": texts[original_idx],
                    "metadata": metadatas[original_idx],
                    "distance": float(distance)
                })
        # If no close matches, take the best one anyway
        if not matched_chunks and len(I[0]) > 0:
            best_idx = I[0][0]
            original_idx = filtered_idxs[best_idx]
            matched_chunks.append({
                "text": texts[original_idx],
                "metadata": metadatas[original_idx],
                "distance": float(D[0][0])
            })
        print(f"Retrieved {len(matched_chunks)} chunks")
        print(f"Pages: {[chunk['metadata'].get('page_numbers', []) for chunk in matched_chunks]}")
        print(f"Element types: {[chunk['metadata'].get('element_category', 'unknown') for chunk in matched_chunks]}")
        state.retrieved_chunks = matched_chunks
        return state
 
    except Exception as e:
        state.error = str(e)
        raise e