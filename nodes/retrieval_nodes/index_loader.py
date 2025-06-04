import faiss
import pickle
from state import QueryState

def index_loader(state: QueryState) -> QueryState:
    try:
        # Load FAISS index
        faiss_index = faiss.read_index(state.index_path)

        # Load metadata
        with open(state.metadata_path, "rb") as f:
            metadata_bundle = pickle.load(f)

        texts = metadata_bundle.get("texts", [])
        metadatas = metadata_bundle.get("metadatas", [])

        # Update state
        return state.copy(update={
            "faiss_index": faiss_index,
            "loaded_metadatas": {
                "texts": texts,
                "metadatas": metadatas
            }
        })

    except Exception as e:
        state.error = str(e)
        raise e
