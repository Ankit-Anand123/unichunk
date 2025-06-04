from sentence_transformers import SentenceTransformer
from state import IngestionState
import faiss
import numpy as np
import pickle
import os

def embedding_node(state: IngestionState) -> IngestionState:
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [chunk["content"] for chunk in state.all_chunks]
        metadatas = [{
            "section_title": chunk.get("section_title"),
            "page_numbers": chunk.get("page_numbers"),
            "chunk_id": chunk.get("chunk_id"),
            "coordinates": chunk.get("coordinates"),
            "chunk_type": chunk.get("chunk_type") ,
            "normalization_factors": chunk.get("normalization_factors")
        } for chunk in state.all_chunks]

        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        os.makedirs(state.index_folder, exist_ok=True)
        faiss.write_index(index, os.path.join(state.index_folder, "faiss.index"))
        with open(os.path.join(state.index_folder, "metadatas.pkl"), "wb") as f:
            pickle.dump({
                "texts": texts,
                "metadatas": metadatas
            }, f)

        # Update state for debugging if needed
        return state.copy(update={
            "embeddings": embeddings,
            "texts": texts,
            "metadatas": metadatas
        })
    except Exception as e:
        state.error = str(e)
        raise e
