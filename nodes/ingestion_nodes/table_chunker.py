from state import IngestionState
from tqdm import tqdm

def table_chunker(state: IngestionState) -> IngestionState:
    """
    Process table elements and create table chunks.
    """
    try:
        table_chunks = []

        for i, el in tqdm(enumerate(state.elements), total=len(state.elements)):
            if el.category == "Table" and hasattr(el, "text") and el.text.strip():
                meta = el.metadata.to_dict()
                table_chunks.append({
                    "section_title": f"Table_{i+1}",
                    "content": el.text.strip(),
                    "page_numbers": [meta.get("page_number")],
                    "coordinates": meta.get("coordinates"),
                    "chunk_id": f"table_{i}",
                    "chunk_type": "table" 
                })

        if state.table_chunks is None:
            state.table_chunks = []
        state.table_chunks.extend(table_chunks)
        return state
    except Exception as e:
        state.error = str(e)
        raise e
