import re
import json
import fitz
import base64
from ollama import chat

def clean_text(text):
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()

def extract_json(text):
    """
    Extract the first valid JSON object from text output.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except Exception:
        pass
    return {}

def convert_base64_to_bytes(base64_str):
    return base64.b64decode(base64_str)

def query_llm_with_images(image_bytes_list, model, prompt):
    response = chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": image_bytes_list
        }]
    )
    return response["message"]["content"]

def parse_coordinates(coords):
    """
    Convert various coordinate formats into a list of fitz.Rect objects.
    Supports nested lists of quadrilaterals like:
    [
      [ [x0, y0], [x1, y1], [x2, y2], [x3, y3] ],
      ...
    ]
    """
    rects = []
    
    if coords is None:
        return rects
    
    try:
        # Handle the format from your chunker: {"points": [...], "system": "...", ...}
        if isinstance(coords, dict) and "points" in coords:
            points = coords["points"]
            
            if isinstance(points, list) and len(points) >= 4:
                # Extract x, y coordinates
                xs = []
                ys = []
                
                for point in points:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        xs.append(float(point[0]))
                        ys.append(float(point[1]))
                
                if len(xs) >= 4 and len(ys) >= 4:
                    # Create bounding rectangle
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    # Ensure rectangle has some area
                    if max_x > min_x and max_y > min_y:
                        rect = fitz.Rect(min_x, min_y, max_x, max_y)
                        if rect.is_valid and not rect.is_empty:
                            rects.append(rect)
        
        # Handle legacy format: list of coordinate lists
        elif isinstance(coords, list):
            for coord_group in coords:
                if coord_group is None:
                    continue
                    
                if isinstance(coord_group, list) and len(coord_group) >= 4:
                    # Handle list of [x, y] pairs
                    if all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in coord_group):
                        try:
                            xs = [float(pt[0]) for pt in coord_group]
                            ys = [float(pt[1]) for pt in coord_group]
                            
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            
                            if max_x > min_x and max_y > min_y:
                                rect = fitz.Rect(min_x, min_y, max_x, max_y)
                                if rect.is_valid and not rect.is_empty:
                                    rects.append(rect)
                        except (ValueError, TypeError):
                            continue
                            
                elif isinstance(coord_group, dict) and "points" in coord_group:
                    # Recursive call for nested dict format
                    nested_rects = parse_coordinates(coord_group)
                    rects.extend(nested_rects)
    
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        print(f"Coordinates format: {type(coords)}")
        if isinstance(coords, dict):
            print(f"Keys: {coords.keys()}")
    
    return rects

def validate_and_clamp_rectangle(rect, page_rect):
    """
    Validate rectangle and clamp to page bounds.
    """
    if not rect or not rect.is_valid or rect.is_empty:
        return None
    
    # Get page dimensions
    page_width = page_rect.width
    page_height = page_rect.height
    
    # Clamp coordinates to page bounds
    x0 = max(0, min(rect.x0, page_width - 1))
    y0 = max(0, min(rect.y0, page_height - 1))
    x1 = max(x0 + 5, min(rect.x1, page_width))  # Ensure minimum width of 5
    y1 = max(y0 + 5, min(rect.y1, page_height))  # Ensure minimum height of 5
    
    clamped_rect = fitz.Rect(x0, y0, x1, y1)
    
    # Final validation
    if clamped_rect.width < 5 or clamped_rect.height < 5:
        return None
    
    return clamped_rect

def debug_coordinates(coords, chunk_id="unknown"):
    """
    Debug function to understand coordinate structure.
    """
    print(f"\n=== DEBUG COORDINATES for {chunk_id} ===")
    print(f"Type: {type(coords)}")
    if isinstance(coords, dict):
        print(f"Keys: {list(coords.keys())}")
        if "points" in coords:
            points = coords["points"]
            print(f"Points type: {type(points)}")
            print(f"Points length: {len(points) if isinstance(points, list) else 'N/A'}")
            if isinstance(points, list) and len(points) > 0:
                print(f"First point: {points[0]}")
                print(f"Last point: {points[-1]}")
    elif isinstance(coords, list):
        print(f"List length: {len(coords)}")
        if len(coords) > 0:
            print(f"First item type: {type(coords[0])}")
            print(f"First item: {coords[0]}")
    print("=" * 50)