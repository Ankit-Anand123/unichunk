from state import IngestionState
from tools import clean_text
from tqdm import tqdm

def text_chunker(state: IngestionState) -> IngestionState:
    """
    Process text-based elements and create text chunks with serialized coordinates.
    """
    try:
        chunk_size = 1000
        grouped = []
        current = {
            "section_title": "Untitled", 
            "content": "", 
            "page_numbers": [], 
            "coordinates": []
        }

        for el in tqdm(state.elements, desc="Processing Elements"):
            text = clean_text(el.text or "")
            page = el.metadata.page_number or 0
            coords = el.metadata.coordinates

            if el.category == "Title":
                if current["content"]:
                    grouped.append(current)
                current = {
                    "section_title": text,
                    "content": "",
                    "page_numbers": [page],
                    "coordinates": []
                }
                if coords:
                    current["coordinates"].append(coords)

            elif el.category in ["NarrativeText", "ListItem", "FigureCaption"]:
                current["content"] += text + " "
                if page not in current["page_numbers"]:
                    current["page_numbers"].append(page)
                if coords:
                    current["coordinates"].append(coords)

        if current["content"]:
            grouped.append(current)

        chunks = []
        for section in grouped:
            words = section["content"].split()
            
            # Serialize coordinates properly
            serialized_coords = []
            for coord in section["coordinates"]:
                if coord and hasattr(coord, 'points'):
                    points = coord.points
                    serialized_points = [(float(x), float(y)) for (x, y) in points]
                    serialized_coords.append(serialized_points)

            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunks.append({
                    "section_title": section["section_title"],
                    "content": " ".join(chunk_words),
                    "page_numbers": section["page_numbers"],
                    "coordinates": serialized_coords, 
                    "chunk_id": f"{section['section_title'][:30]}_{i//chunk_size}",
                    "chunk_type": "text"
                })

        if state.text_chunks is None:
            state.text_chunks = []
        state.text_chunks.extend(chunks)
        return state

    except Exception as e:
        state.error = str(e)
        raise e
