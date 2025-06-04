from state import IngestionState
from tools import clean_text
from tqdm import tqdm

def text_chunker(state: IngestionState) -> IngestionState:
    """
    Process text-based elements with ONE element per chunk for precise coordinate mapping.
    This ensures each chunk has exactly one set of coordinates.
    """
    try:
        chunks = []
        
        for idx, el in enumerate(tqdm(state.elements, desc="Processing Elements")):
            text = clean_text(el.text or "")
            if not text or len(text.strip()) < 10:  # Skip very short texts
                continue
                
            page = el.metadata.page_number or 0
            coords = el.metadata.coordinates
            
            # Only process text elements
            if el.category not in ["Title", "NarrativeText", "ListItem", "FigureCaption"]:
                continue
            
            # Store coordinates exactly as they come from unstructured
            original_coords = None
            if coords and hasattr(coords, 'points'):
                # Store the raw coordinate data
                original_coords = {
                    "points": coords.points,
                    "system": getattr(coords, 'system', None),
                    "layout_width": getattr(coords, 'layout_width', None),
                    "layout_height": getattr(coords, 'layout_height', None)
                }
            
            # Create ONE chunk per element
            chunk_id = f"{el.category}_{idx}_page_{page}"
            
            chunks.append({
                "section_title": f"{el.category}: {text[:50]}..." if len(text) > 50 else f"{el.category}: {text}",
                "content": text,
                "page_numbers": [page],  # Single page only
                "coordinates": original_coords,  # Single coordinate set
                "chunk_id": chunk_id,
                "chunk_type": "text",
                "element_category": el.category,
                "element_index": idx,
                "text_length": len(text)
            })

        if state.text_chunks is None:
            state.text_chunks = []
        state.text_chunks.extend(chunks)
        
        print(f"Created {len(chunks)} text chunks")
        return state

    except Exception as e:
        state.error = str(e)
        raise e