from state import IngestionState

def chunk_aggregator(state: IngestionState) -> IngestionState:
    try:
        all_chunks = (state.text_chunks or []) + (state.table_chunks or []) + (state.image_chunks or [])

        if state.all_chunks is None:
            state.all_chunks = []
        state.all_chunks.extend(all_chunks)  
        
        return state
    except Exception as e:
        state.error = str(e)
        raise e