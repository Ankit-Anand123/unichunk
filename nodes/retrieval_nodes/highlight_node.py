from state import QueryState

def highlight_node(state: QueryState) -> QueryState:
    try:
        highlights = []
        
        print(f"Processing {len(state.retrieved_chunks)} chunks for highlighting")
        
        for i, chunk in enumerate(state.retrieved_chunks):
            meta = chunk.get("metadata", {})
            page_numbers = meta.get("page_numbers", [])
            coordinates = meta.get("coordinates", None)
            chunk_id = meta.get("chunk_id", f"chunk_{i}")
            
            print(f"Chunk {i}: {chunk_id}")
            print(f"  Pages: {page_numbers}")
            print(f"  Has coordinates: {coordinates is not None}")
            
            # Only store if coordinates exist and are valid
            if coordinates is not None:
                # Since we now have one element per chunk, we should have exactly one page
                if len(page_numbers) == 1:
                    highlight_entry = {
                        "page_numbers": page_numbers,
                        "coordinates": coordinates,  # Store original format
                        "chunk_id": chunk_id,
                        "element_category": meta.get("element_category", "unknown"),
                        "distance": chunk.get("distance", 999)  # For debugging
                    }
                    highlights.append(highlight_entry)
                    print(f"  Added highlight for page {page_numbers[0]}")
                else:
                    print(f"  Skipped: multiple pages {page_numbers}")
            else:
                print(f"  Skipped: no coordinates")

        print(f"Total highlights created: {len(highlights)}")
        
        return state.copy(update={
            "highlighted_pages": highlights
        })

    except Exception as e:
        print(f"Error in highlight_node: {e}")
        state.error = str(e)
        raise e