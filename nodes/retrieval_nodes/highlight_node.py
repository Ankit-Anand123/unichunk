from state import QueryState

def highlight_node(state: QueryState) -> QueryState:
    try:
        highlights = []
        
        for chunk in state.retrieved_chunks:
            meta = chunk.get("metadata", {})
            page_numbers = meta.get("page_numbers", [])
            coordinates = meta.get("coordinates", None)

            # Only store if coordinates exist
            if coordinates is not None:
                highlight_entry = {
                    "page_numbers": page_numbers,
                    "coordinates": coordinates,
                    "chunk_id": meta.get("chunk_id")
                }
                highlights.append(highlight_entry)

        return state.copy(update={
            "highlighted_pages": highlights
        })

    except Exception as e:
        state.error = str(e)
        raise e