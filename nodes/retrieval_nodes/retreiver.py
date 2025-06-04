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

        # Unpack loaded metadatas (texts and metadatas)
        texts = state.loaded_metadatas["texts"]
        metadatas = state.loaded_metadatas["metadatas"]

        # Apply modality filtering
        if state.query_modality == "text_only":
            filtered_idxs = [i for i, meta in enumerate(metadatas) if meta.get("chunk_type") == "text"]
        elif state.query_modality == "table_required":
            filtered_idxs = [i for i, meta in enumerate(metadatas) if meta.get("chunk_type") == "table"]
        elif state.query_modality == "visual_required":
            filtered_idxs = [i for i, meta in enumerate(metadatas) if meta.get("chunk_type") == "visual"]
        else:
            filtered_idxs = list(range(len(metadatas)))  # No filtering, use all

        # If nothing matched, fall back to using all chunks
        if not filtered_idxs:
            filtered_idxs = list(range(len(metadatas)))

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

        query_emb = np.array(state.query_embedding, dtype=np.float32).reshape(1, -1)
        D, I = tmp_index.search(query_emb, k=min(10, embeddings.shape[0]))

        matched_chunks = [
            {
                "text": texts[filtered_idxs[idx]],
                "metadata": metadatas[filtered_idxs[idx]]
            }
            for idx in I[0]
        ]
        
        state.retrieved_chunks = matched_chunks
        return state

    except Exception as e:
        state.error = str(e)
        raise e
