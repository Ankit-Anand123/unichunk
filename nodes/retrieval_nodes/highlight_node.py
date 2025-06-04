import fitz
from state import QueryState

def highlight_node(state: QueryState) -> QueryState:
    try:
        highlights = []

        for chunk in state.retrieved_chunks:
            page_numbers = chunk.get("page_numbers")
            coordinates = chunk.get("coordinates")
            normalization_factors = chunk.get("normalization_factors")

            if not coordinates:
                continue

            # Visual chunks: already normalized fitz.Rect
            if isinstance(coordinates, fitz.Rect):
                highlights.append({
                    "page_numbers": page_numbers,
                    "coordinates": coordinates
                })

            # Table chunks: normalized dict with points & layout
            elif isinstance(coordinates, dict) and "points" in coordinates:
                points = coordinates["points"]
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                highlights.append({
                    "page_numbers": page_numbers,
                    "coordinates": (x0, y0, x1, y1)
                })

            # Text chunks: deeply nested list of points
            elif isinstance(coordinates, list):
                for group in coordinates:
                    xs = [float(pt[0]) for pt in group]
                    ys = [float(pt[1]) for pt in group]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    highlights.append({
                        "page_numbers": page_numbers,
                        "coordinates": (x0, y0, x1, y1)
                    })

            else:
                print(f"Skipping unknown coordinate format: {type(coordinates)}")

        state.highlighted_pages = highlights
        return state

    except Exception as e:
        state.error = str(e)
        raise e