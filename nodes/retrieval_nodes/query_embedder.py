from state import QueryState
from sentence_transformers import SentenceTransformer
import numpy as np

def query_embedder(state: QueryState) -> QueryState:
    try:
        # Load embedding model
        model = SentenceTransformer(state.embedding_model)

        # Generate embedding for query
        query_embedding = model.encode(
            state.query,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Ensure shape compatibility with FAISS
        query_embedding = np.array([query_embedding])

        return state.copy(update={
            "query_embedding": query_embedding
        })

    except Exception as e:
        state.error = str(e)
        raise e