import numpy as np
from state import QueryState
import faiss
import pickle
from sentence_transformers import SentenceTransformer

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
            filtered = [(t, m) for t, m in zip(texts, metadatas) if m.get("chunk_type") == "text"]
        elif state.query_modality == "table_required":
            filtered = [(t, m) for t, m in zip(texts, metadatas) if m.get("chunk_type") == "table"]
        elif state.query_modality == "visual_required":
            filtered = [(t, m) for t, m in zip(texts, metadatas) if m.get("chunk_type") == "visual"]
        else:
            filtered = list(zip(texts, metadatas))

        # Fallback safeguard
        if not filtered:
            filtered = list(zip(texts, metadatas))

        filtered_texts = [t for t, _ in filtered]
        filtered_metadatas = [m for _, m in filtered]

        # Compute embeddings again for filtered chunks
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(filtered_texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        tmp_index = faiss.IndexFlatL2(dim)
        tmp_index.add(embeddings)

        query_emb = np.array(state.query_embedding, dtype=np.float32).reshape(1, -1)
        D, I = tmp_index.search(query_emb, k=min(10, embeddings.shape[0]))

        matched_chunks = []
        for idx in I[0]:
            chunk = {
                "text": filtered_texts[idx],
                **filtered_metadatas[idx]
            }
            matched_chunks.append(chunk)

        state.retrieved_chunks = matched_chunks
        return state

    except Exception as e:
        state.error = str(e)
        raise e