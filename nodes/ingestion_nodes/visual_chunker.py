
from state import IngestionState
from tqdm import tqdm
from tools import convert_base64_to_bytes, query_llm_with_images

def visual_chunker(state: IngestionState) -> IngestionState:
    try:
        vlm_prompt = state.vlm_prompt
        model = state.vlm_model or "qwen2.5vl:7b"

        visual_chunks = []
        filtered_elements = [el for el in state.partitioned_elements if el.category == "Image"]

        for idx, el in enumerate(tqdm(filtered_elements, desc="Analyzing Images")):
            image_base64 = getattr(el.metadata, "image_base64", None)
            if image_base64 is None:
                continue
            image_bytes = convert_base64_to_bytes(image_base64)
            result = query_llm_with_images([image_bytes], model=model, prompt=vlm_prompt)
            visual_chunks.append({
                    "section_title": el.text,
                    "content": result,
                    "page_numbers": el.metadata.page_number,
                    "coordinates": el.metadata.coordinates,
                    "chunk_id": f"visual_page_{el.metadata.page_number}",
                    "chunk_type": "visual"
                })

        if state.image_chunks is None:
            state.image_chunks = []
        state.image_chunks.extend(visual_chunks)  
        return state
    except Exception as e:
        state.error = str(e)
        raise e

