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

    if isinstance(coords, fitz.Rect):
        rects.append(coords)

    elif isinstance(coords, dict) and "points" in coords:
        points = coords["points"]
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))

    elif isinstance(coords, list):
        for group in coords:
            xs = [float(pt[0]) for pt in group]
            ys = [float(pt[1]) for pt in group]
            rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))

    else:
        print(f"Unknown coordinate format: {type(coords)}")

    return rects

